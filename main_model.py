import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        # physics hooks (dataset-specific loss is attached by dataset-specific model)
        self.use_physics = False
        self.lambda_phys = 0.0
        self.mean = None
        self.std = None
        self.physics_loss_fn = None
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device, dtype=torch.float32) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        # If a dataset-specific physics loss is attached, compute and add it
        if getattr(self, "use_physics", False) and (self.physics_loss_fn is not None):
            # current_alpha is alpha_bar_t with shape (B,1,1)
            sqrt_alpha = current_alpha.sqrt()
            sqrt_one_minus_alpha = (1.0 - current_alpha).sqrt()

            # reconstruct x0 estimate: shape (B,K,L)
            x_hat = (noisy_data - sqrt_one_minus_alpha * predicted) / (sqrt_alpha + 1e-8)

            try:
                phys_loss = self.physics_loss_fn(x_hat)
                loss = loss + float(self.lambda_phys) * phys_loss
            except Exception:
                # if physics loss fails, skip it gracefully
                pass

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device, dtype=torch.float32)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(self.device, dtype=torch.float32)
        observed_tp = batch["timepoints"].to(self.device, dtype=torch.float32)
        gt_mask = batch["gt_mask"].to(self.device, dtype=torch.float32)
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device, dtype=torch.float32)

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(self.device, dtype=torch.float32)
        observed_tp = batch["timepoints"].to(self.device, dtype=torch.float32)
        gt_mask = batch["gt_mask"].to(self.device, dtype=torch.float32)

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Fluid_Kaggle(CSDI_base):
    def __init__(
        self,
        config,
        device,
        target_dim,
        use_physics=False,
        lambda_phys=1.0,
        mean=None,
        std=None,
    ):
        """
        Dataset-specific wrapper for the Kaggle fluid flow CSV data (processed by
        `dataset_flow.read_laminar_flow`). Accepts optional physics args so
        training script can pass `use_physics`, `lambda_phys`, `mean`, and `std`.
        When `use_physics` is True this will attempt to attach a dataset-specific
        physics loss function; if unavailable the physics hook is disabled.
        """
        super(CSDI_Fluid_Kaggle, self).__init__(target_dim, config, device)
        model_cfg = config.get("model", {})
        physics_cfg = model_cfg.get("physics", {})

        use_phys = bool(use_physics or model_cfg.get("use_physics", model_cfg.get("use_physics_loss", 0)))
        lambda_p = float(lambda_phys if lambda_phys is not None else model_cfg.get("lambda_phys", model_cfg.get("physics_loss_weight", 0.0)))

        if use_phys:
            try:
                # Try to import a dataset-specific physics module. If not present,
                # fall back to disabling physics.
                from physics import physics_fluid

                dt = float(physics_cfg.get("dt", 0.1))

                mean_t = None if mean is None else torch.tensor(mean, dtype=torch.float32, device=device)
                std_t = None if std is None else torch.tensor(std, dtype=torch.float32, device=device)

                def _make_phys_fn(mean_tensor, std_tensor):
                    def _fn(x_hat):
                        m = mean_tensor if mean_tensor is not None else self.mean
                        s = std_tensor if std_tensor is not None else self.std
                        # Delegate to dataset-specific implementation. Keep placeholder
                        # name `physics_loss_fn` to match other dataset hooks.
                        return physics_fluid.physics_loss_fn(x_hat, m, s, dt=dt)

                    return _fn

                self.physics_loss_fn = _make_phys_fn(mean_t, std_t)
                self.use_physics = True
                self.lambda_phys = lambda_p
                if mean_t is not None:
                    self.mean = mean_t
                if std_t is not None:
                    self.std = std_t
            except Exception:
                # If import or setup fails, disable physics gracefully.
                self.physics_loss_fn = None
                self.use_physics = False
                self.lambda_phys = 0.0
        else:
            self.physics_loss_fn = None
            self.use_physics = False
            self.lambda_phys = 0.0

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(self.device, dtype=torch.float32)
        observed_tp = batch["timepoints"].to(self.device, dtype=torch.float32)
        gt_mask = batch["gt_mask"].to(self.device, dtype=torch.float32)

        # Permute to (B, K, L) like other dataset wrappers
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Traffic(CSDI_base):
    def __init__(
        self,
        config,
        device,
        target_dim=6,
        use_physics=False,
        lambda_phys=1.0,
        mean=None,
        std=None,
    ):
        """
        Dataset-specific wrapper for traffic. Accepts optional physics args so training script
        can pass `use_physics`, `lambda_phys`, `mean`, and `std` directly.
        """
        super(CSDI_Traffic, self).__init__(target_dim, config, device)
        model_cfg = config.get("model", {})
        physics_cfg = model_cfg.get("physics", {})

        # determine whether physics is enabled (explicit arg takes precedence)
        use_phys = bool(use_physics or model_cfg.get("use_physics", model_cfg.get("use_physics_loss", 0)))
        lambda_p = float(lambda_phys if lambda_phys is not None else model_cfg.get("lambda_phys", model_cfg.get("physics_loss_weight", 0.0)))

        if use_phys:
            try:
                from physics import physics_traffic

                pos_idx = int(physics_cfg.get("pos_index", physics_cfg.get("pos_idx", 0)))
                vel_idx = int(physics_cfg.get("vel_index", physics_cfg.get("vel_idx", 2)))
                acc_idx = int(physics_cfg.get("acc_index", physics_cfg.get("acc_idx", 3)))
                dt = float(physics_cfg.get("dt", 0.1))

                mean_t = None if mean is None else torch.tensor(mean, dtype=torch.float32, device=device)
                std_t = None if std is None else torch.tensor(std, dtype=torch.float32, device=device)

                # closure that will use mean/std from closure or fallback to self.mean/self.std
                def _make_phys_fn(mean_tensor, std_tensor):
                    def _fn(x_hat):
                        m = mean_tensor if mean_tensor is not None else self.mean
                        s = std_tensor if std_tensor is not None else self.std
                        return physics_traffic.physics_loss_soft(
                            x_hat, m, s, pos_idx=pos_idx, vel_idx=vel_idx, acc_idx=acc_idx, dt=dt
                        )

                    return _fn

                self.physics_loss_fn = _make_phys_fn(mean_t, std_t)
                self.use_physics = True
                self.lambda_phys = lambda_p
                if mean_t is not None:
                    self.mean = mean_t
                if std_t is not None:
                    self.std = std_t
            except Exception:
                self.physics_loss_fn = None
                self.use_physics = False
                self.lambda_phys = 0.0
        else:
            self.physics_loss_fn = None

            self.use_physics = False
            self.lambda_phys = 0.0

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(self.device, dtype=torch.float32)
        observed_tp = batch["timepoints"].to(self.device, dtype=torch.float32)
        gt_mask = batch["gt_mask"].to(self.device, dtype=torch.float32)

        # permute from (B, L, K) -> (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_RBC(CSDI_base):
    def __init__(
        self,
        config,
        device,
        target_dim,
        use_physics=False,
        lambda_phys=1.0,
        mean=None,
        std=None,
    ):
        """
        Dataset-specific wrapper for Rayleigh-Benard (RBC) data. Accepts optional
        physics args so training script can pass `use_physics`, `lambda_phys`, `mean`, and `std`.
        """
        super(CSDI_RBC, self).__init__(target_dim, config, device)
        model_cfg = config.get("model", {})
        physics_cfg = model_cfg.get("physics", {})

        use_phys = bool(use_physics or model_cfg.get("use_physics", model_cfg.get("use_physics_loss", 0)))
        lambda_p = float(lambda_phys if lambda_phys is not None else model_cfg.get("lambda_phys", model_cfg.get("physics_loss_weight", 0.0)))

        if use_phys:
            try:
                # attempt to import dataset-specific physics loss; may not exist yet
                from physics import physics_rbc

                # indices and dt are dataset-dependent; allow config override
                dt = float(physics_cfg.get("dt", 0.1))

                mean_t = None if mean is None else torch.tensor(mean, dtype=torch.float32, device=device)
                std_t = None if std is None else torch.tensor(std, dtype=torch.float32, device=device)

                def _make_phys_fn(mean_tensor, std_tensor):
                    def _fn(x_hat):
                        m = mean_tensor if mean_tensor is not None else self.mean
                        s = std_tensor if std_tensor is not None else self.std
                        return physics_rbc.physics_loss_fn(x_hat, m, s, dt=dt)

                    return _fn

                self.physics_loss_fn = _make_phys_fn(mean_t, std_t)
                self.use_physics = True
                self.lambda_phys = lambda_p
                if mean_t is not None:
                    self.mean = mean_t
                if std_t is not None:
                    self.std = std_t
            except Exception:
                self.physics_loss_fn = None
                self.use_physics = False
                self.lambda_phys = 0.0
        else:
            self.physics_loss_fn = None
            self.use_physics = False
            self.lambda_phys = 0.0

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(self.device, dtype=torch.float32)
        observed_tp = batch["timepoints"].to(self.device, dtype=torch.float32)
        gt_mask = batch["gt_mask"].to(self.device, dtype=torch.float32)

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )



class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
