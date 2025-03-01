"""Set of functions to specify the trained parameters at different phase."""


def last_only(model, log=print):
    if hasattr(model, "module"):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True
    if hasattr(model, "scale_head"):
        if model.scale_head is not None:
            for p in model.scale_head.parameters():
                p.requires_grad = True


def group_last_only(model, log=print):
    if hasattr(model, "module"):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer_group.parameters():
        p.requires_grad = True
    for p in model.group_projection.parameters():
        p.requires_grad = False
    if hasattr(model, "scale_head"):
        if model.scale_head is not None:
            for p in model.scale_head.parameters():
                p.requires_grad = True


def warm_only(model, log=print):

    is_seg = str(model.features).upper().startswith("SEGFORMER")

    if not is_seg:
        is_v3 = str(model.features.base).upper().startswith("DEEPLABV3")

    if is_seg:
        aspp_params = []
    elif is_v3:
        aspp_params = model.features.base.aspp.parameters()
    else:
        aspp_params = [
            model.features.base.aspp.c0.weight,
            model.features.base.aspp.c0.bias,
            model.features.base.aspp.c1.weight,
            model.features.base.aspp.c1.bias,
            model.features.base.aspp.c2.weight,
            model.features.base.aspp.c2.bias,
            model.features.base.aspp.c3.weight,
            model.features.base.aspp.c3.bias,
        ]

    if hasattr(model, "module"):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    for p in aspp_params:
        p.requires_grad = True

    if hasattr(model, "scale_head"):
        if model.scale_head is not None:
            for p in model.scale_head.parameters():
                p.requires_grad = True


def group_only(model, log=print):
    if hasattr(model, "module"):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer_group.parameters():
        p.requires_grad = False

    if hasattr(model, "scale_head"):
        if model.scale_head is not None:
            for p in model.scale_head.parameters():
                p.requires_grad = False
    for p in model.group_projection.parameters():
        p.requires_grad = True


def joint(model, log=print):
    if hasattr(model, "module"):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True

    if hasattr(model, "scale_head"):
        if model.scale_head is not None:
            for p in model.scale_head.parameters():
                p.requires_grad = True


def group_joint(model, joint_no_proto, joint_last, log=print):
    if hasattr(model, "module"):
        model = model.module

    if not joint_last:
        for p in model.features.parameters():
            p.requires_grad = True
        for p in model.add_on_layers.parameters():
            p.requires_grad = True
        if joint_no_proto:
            model.prototype_vectors.requires_grad = False
        else:
            model.prototype_vectors.requires_grad = True
    else:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.add_on_layers.parameters():
            p.requires_grad = False
        model.prototype_vectors.requires_grad = False

    for p in model.last_layer_group.parameters():
        p.requires_grad = True
    for p in model.group_projection.parameters():
        p.requires_grad = True

    if hasattr(model, "scale_head"):
        if model.scale_head is not None:
            for p in model.scale_head.parameters():
                p.requires_grad = True
