from models.lkunetComplex import lkunetComplex
from models.voxelMorphComplex import voxelMorphComplex
from models.transMorphComplex import transMorphComplex
from models.memWarpComplex import memWarpComplex
from models.lapWarpComplex import lapWarpComplex

def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None

    if 'lkunetComplex' in model_name:
        model = lkunetComplex(**nkwargs)
    elif 'voxelMorphComplex' in model_name:
        model = voxelMorphComplex(**nkwargs)
    elif 'transMorphComplex' in model_name:
        model = transMorphComplex(**nkwargs)
    elif 'memWarpComplex' in model_name:
        model = memWarpComplex(**nkwargs)
    elif 'lapWarpComplex' in model_name:
        model = lapWarpComplex(**nkwargs)
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model