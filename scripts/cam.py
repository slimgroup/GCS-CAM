from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Get CAM heatmaps 
def find_cam(model,input_image_tensor, cam_method = "ScoreCAM"):
  
    input_tensor = input_image_tensor

    def vit_reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

        # Similar to CNN case, bring channels to first dimension
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    # Choose a layer except the last attention layer. Generally penultimate layer gives good results.
    target_layers = [model.blocks[-2].norm1]
    model_transform = vit_reshape_transform
    
    cam =  ScoreCAM(model, target_layers=target_layers, use_cuda=True, reshape_transform = model_transform)
    targets = [ClassifierOutputTarget(1)]  

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam