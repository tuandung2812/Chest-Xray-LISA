from .lxmert_0_explained import LXMERT0Explained


def build_lxmert_0_explained(
    use_lrp
) -> LXMERT0Explained:
    model = LXMERT0Explained(
        use_lrp=use_lrp
    )
    return model


lxmert_0_explained_model_registry = {
    "lxmert_0": build_lxmert_0_explained,
} 