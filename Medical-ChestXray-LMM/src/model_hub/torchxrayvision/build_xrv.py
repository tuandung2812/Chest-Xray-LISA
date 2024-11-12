from .modeling import PSPNet


def build_xrv_pspnet(
        
):
    xrv = PSPNet()
    xrv.eval()
    return xrv


xrv_model_registry = {
    "xrv_pspnet": build_xrv_pspnet,
}
