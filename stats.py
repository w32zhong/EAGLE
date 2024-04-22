class SpeculativeStats():
    def __init__(self, name):
        self.name = name
        self.shapes_full_x = []
        self.shapes_full_y = []
        self.samples_full = []
        self.samples_draft = []
        self.samples_e2e = []
        self.samples_new_toks = []
        self.samples_toks = []
        self.n_trials = 1

    def add_json_sample(self, j):
        model_stats = j['output_trials'][0]['_model_stats']
        self.shapes_full_x += model_stats['cur_len']
        self.shapes_full_y += model_stats['past_len']
        self.samples_full += model_stats['ea_generate verify forward']
        self.samples_draft += model_stats['tree_drafting']
        self.samples_e2e += model_stats['ea_generate iteration']
        self.samples_new_toks += model_stats['#new tokens per iteration']
        out_tokens = len(j['output_trials'][0]["out_tokens"])
        self.samples_toks.append(out_tokens)

    def report(self):
        return dict(
            name=self.name,
            tot_verify=len(self.samples_full),
            avg_forward_ms_per_verify=sum(self.samples_full) / len(self.samples_full),
            avg_new_tokens_per_verify=sum(self.samples_new_toks) / len(self.samples_full),
            mean_verify_shape_x=sum(self.shapes_full_x) / len(self.shapes_full_x),
            mean_verify_shape_y=sum(self.shapes_full_y) / len(self.shapes_full_y),
            avg_forward_ms_per_draft=sum(self.samples_draft) / len(self.samples_draft),
            avg_forward_ms_per_token=sum(self.samples_e2e) / sum(self.samples_toks)
        )
