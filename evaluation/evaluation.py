import cv2
import numpy as np
import skimage.util
import skimage.metrics
import skimage.color

def psnr(InputRGB, OutputRGB):
    return cv2.PSNR(InputRGB, OutputRGB)

def ssim(InputRGB, OutputRGB):
    return skimage.metrics.structural_similarity(InputRGB, OutputRGB, channel_axis = 2)

def deltaE(InputLAB, OutputLAB):
    return np.mean(skimage.color.deltaE_cie76(InputLAB, OutputLAB))

def evaluation_metrics(Original, Out):
    psnr_out = psnr(Original, Out)
    ssim_out = ssim(Original, Out)
    deltaE_out = deltaE(Original, Out)
    cid_out = CID(Original, Out)
    return psnr_out, ssim_out, deltaE_out, cid_out
  
  
  
# Input Image IMG Lab, S is patch dim, Step is stride, Ds is downsampling

def image_generation(ImgLAB, S, Step, ds): # For Factorial Combination Model
    scaledS = int(S/ds)
    scaleStep = int(Step/ds)
    ds_shape = (int(np.ceil(ImgLAB.shape[1] * 1.0/ds)), int(np.ceil(ImgLAB.shape[0] * 1.0/ds)))
    resized_img = cv2.resize(ImgLAB, ds_shape)
    img_window = skimage.util.view_as_windows(resized_img, (scaledS,scaledS,3), scaleStep)
    num_row, num_col = img_window.shape[0:2]
    return img_window.reshape((num_row * num_col, scaledS, scaledS, 3))
   

def lightness_contrast_multilevel(img1_win, img2_win): # Seems good, take in WINDOWS
    c2 = .1
    sdev_1 = np.std(img1_win[:,:,:,0], axis = (1,2))
    sdev_2 = np.std(img2_win[:,:,:,0], axis = (1,2))
    
    cL = np.mean(np.divide(2 * np.multiply(sdev_1, sdev_2) + c2, np.square(sdev_1) + np.square(sdev_2) + c2)) # Eventually going to average
    return cL
    
    
def lightness_structure_multilevel(img1_win, img2_win, c3 = .1):
    X = img1_win[:,:,:,0] # luminance channels
    Y = img2_win[:,:,:,0]
    
    S = X.shape[1]
    
    sdev_x = np.std(X, axis = (1,2))
    sdev_y = np.std(Y, axis = (1,2))
    
    X_mean_gauss = np.empty(X.shape)
    Y_mean_gauss = np.empty(Y.shape)
    for idx, win in enumerate(X):
        X_mean_gauss[idx] = cv2.GaussianBlur(win, ksize = (S,S), sigmaX = 1)

    for idx, win in enumerate(Y):
        Y_mean_gauss[idx] = cv2.GaussianBlur(win, ksize = (S,S), sigmaX = 1)
        
    X_diff = np.empty(X.shape)
    Y_diff = np.empty(Y.shape)
    
    meanX = np.mean(X_mean_gauss, axis = (1,2))
    meanY = np.mean(Y_mean_gauss, axis = (1,2))
    
    for i in range(X.shape[0]):
        X_diff[i,:,:] = X[i,:,:] - meanX[i]
        Y_diff[i,:,:] = Y[i,:,:] - meanY[i]
    
    base_shape = X_diff.shape
        
    X_diff_vec = X_diff.reshape((base_shape[0], base_shape[1]**2) )
    Y_diff_vec = Y_diff.reshape((base_shape[0], base_shape[1]**2) )
    
    norms = np.multiply(np.linalg.norm(X_diff_vec, axis = 1), np.linalg.norm(Y_diff_vec, axis = 1))
    dotted = np.sum(np.multiply(X_diff_vec, Y_diff_vec), axis = 1)
    
    angles = np.divide(dotted, norms + 2**-52) # Dot Product between each
    sL = np.mean(np.divide(angles + c3, c3 + np.multiply(sdev_x, sdev_y)))
    
    return sL


def CID(InputBGR, OutputBGR, S=15, Step=10, N=3): # InputBGR, OutputBGR (Doesn't matter), Step Size, Patch Size, N= Multilevel order of CID

    
    InputLAB = cv2.cvtColor(InputBGR, cv2.COLOR_BGR2LAB)
    OutputLAB = cv2.cvtColor(OutputBGR, cv2.COLOR_BGR2LAB)
    inp_base = image_generation(InputLAB, S, Step, 1)
    out_base = image_generation(OutputLAB, S, Step, 1)
   
    
    inp_downsampled = []
    out_downsampled = []
    
    #inp_lite = inp_base[:,:,:,0] #luminance channels for fully sampled image
    #out_lite = out_base[:,:,:,0]
    
    #inp_downsampled.append(inp_lite)
    #out_downsampled.append(inp_lite)
    
    for i in range(0,N):
        inp_downsampled.append(image_generation(InputLAB, S, Step,2 ** i )) # Fully Sampled, 2x down, 4x down
        out_downsampled.append(image_generation(OutputLAB, S, Step,2 ** i ))
    
    
   
    
    c_array = np.array([.002, .1, .1, .002, .008]) # Hyperparameters (Scale Weights)
    a_array = np.array([.0448, .2856, .3001])
    
    # Lightness, Chroma, Hue Comparisons
    # Ll(x,y)
    inp_max_down = inp_downsampled[-1]
    
    out_max_down = out_downsampled[-1]
    deltaL = inp_max_down[:,:,:,0] - out_max_down[:,:,:,0] # Lightness Components
    
    
    l_L_gauss = np.empty(deltaL.shape) # Storage for l_L (9) for each patch
    
    Size_down_max = deltaL.shape[1] # Patch size for max downsampled (Should be 3)
    print(Size_down_max)
    
    for idk, win in enumerate(deltaL):
        l_L_gauss[idk] = cv2.GaussianBlur(win, ksize = (Size_down_max,Size_down_max), sigmaX = 1) # K-size is full window (3 for max down)
    
    l_l = np.mean(np.reciprocal(1 + c_array[0] * np.square(np.mean(l_L_gauss, axis=(1,2))))) ** a_array[2] # Ll component of equation 17 for n=3
    
  
    
    
    # Lc(x,y)
    delta_C = np.abs(np.sqrt(np.square(inp_base[:,:,:,1]) + np.square(inp_base[:,:,:,2])) - np.sqrt(np.square(out_base[:,:,:,1]) + np.square(out_base[:,:,:,2])))
    delta_C_gauss = np.empty(delta_C.shape)
    delta_C = np.float64(delta_C)
    
    for idxC, winC in enumerate(delta_C):
        delta_C_gauss[idxC] = cv2.GaussianBlur(winC, ksize = (S,S), sigmaX = 1)
    
    l_C = np.mean(np.reciprocal(1 + c_array[3] * np.square(np.mean(delta_C_gauss,axis=(1,2))))) # 10 Final, per pixel, will eventually average over this
    
    
    
   
    
    
    #l_L
    delta_H = np.sqrt(np.abs(np.square(inp_base[:,:,:,1] - inp_base[:,:,:,1]) + np.square(out_base[:,:,:,2] - out_base[:,:,:,2]) - np.square(delta_C)))
    delta_H = np.float64(delta_H)
    delta_H_gauss = np.empty(delta_H.shape)
    
    for idxH, winH in enumerate(delta_H):
        delta_H_gauss[idxH] = cv2.GaussianBlur(winH, ksize = (S,S), sigmaX = 1)
    
    
    l_H = np.mean(np.reciprocal(1 + c_array[4] * np.square(np.mean(delta_H_gauss,axis=(1,2))))) # 11 Final, per pixel, will eventually average over thisl_H = 
    
    
    #####
    
    #### Lightness- Contrast Comparison (2)
    
   
    cL_list = np.zeros(N) # Should be 3 levels
    sL_list = np.zeros(N)
    #cL_list.append(lightness_contrast_multilevel(inp_lite))
    for i in range(N):
        cL_list[i] = lightness_contrast_multilevel(inp_downsampled[i], out_downsampled[i])
    #### Lghtness-Contrast Multilevel (3)
    
    
    
    for i in range(N):
        sL_list[i] = lightness_structure_multilevel(inp_downsampled[i], out_downsampled[i])
    
    
    cL_sL_prod_exp =np.prod(np.exp(np.multiply(cL_list, sL_list),a_array)) #17, product
    
    
    return 1 - cL_sL_prod_exp * l_l * l_C * l_H
    
    
    
    
    
    
    
    