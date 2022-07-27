"""
Created on Mon Jul 25 2019

@author: stalin
"""

import nibabel as nib
import numpy as np
import cv2
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from PIL import Image

def tumor_extraction(imagen_original_nii, slice):
    direccion = imagen_original_nii + '.gz'
    img = nib.load(direccion)
    img_data = np.asanyarray(img.dataobj)
    a = img_data.shape[2] - 1
    corte = img_data[:,:, a-slice]
    
    normalizedImg = np.zeros((320, 320))                                            #NUEVA
    normalizedImg = cv2.normalize(corte,  normalizedImg, 0, 255, cv2.NORM_MINMAX)   #NUEVA
    
    aux = normalizedImg.reshape((-1, 1))
    aux = np.float32(aux)
    
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(aux, 3, None, criterio, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    aux_out = center[label.flatten()]
    img_kmeans3 = aux_out.reshape((corte.shape))
    
    umbral = center.reshape((-1,1))
    umbral = sorted(umbral)
    valor_umbral = umbral[1][0]
    aux1, mascara = cv2.threshold(img_kmeans3, valor_umbral, 255, cv2.THRESH_BINARY)
    bordes = cv2.Canny(mascara, 10, 10)
    img_erosion = nd.grey_erosion(mascara, size=(9,9))
    img_dilation = nd.grey_dilation(img_erosion, size=(9,9))
    
    bordes2 = cv2.Canny(img_dilation, 10, 10)               #NUEVA
    cor = np.uint8(normalizedImg)                           #NUEVA
    rgb = cv2.cvtColor(cor, cv2.COLOR_GRAY2BGR)             #NUEVA
    bgr = cv2.cvtColor(cor, cv2.COLOR_GRAY2BGR)             #NUEVA
    rgb[:,:,2] = bordes; rgb[:,:,1] = 0 ; rgb[:,:,0] = 0    #NUEVA
    bgr[:,:,2] = bordes2; bgr[:,:,1] = 0 ; bgr[:,:,0] = 0   #NUEVA
    b, g, r = cv2.split(rgb)                                #NUEVA
    bb, gg, rr = cv2.split(bgr)                             #NUEVA
    img_dilation = np.float64(img_dilation/255)             #NUEVA
    
    corte = np.array(corte)
    img_dilation = np.array(img_dilation)   #MASCARA FINAL
    tumor = corte*img_dilation      #TUMOR SEGMENTADO 31-17

    bordes = np.array(1-bordes/255)                                 #NUEVA
    bordet = np.array(1- bordes2/255)                               #NUEVA
    borde = corte * bordes                                          #NUEVA
    bordet = corte * bordet                                         #NUEVA
    borde = cv2.convertScaleAbs(borde, alpha=(255/corte.max()))     #NUEVA
    bordet = cv2.convertScaleAbs(bordet, alpha=(255/corte.max()))   #NUEVA
    borde = borde | r                                               #NUEVA
    bordet = bordet | rr                                            #NUEVA

    intensidades = []
    for i in range(tumor.shape[0]):
        for j in range(tumor.shape[1]):
            if tumor[i,j] != 0:
                intensidades.append(tumor[i,j])           
    int_min = min(intensidades)
    #SEGMENTACION TUMOR COMPLETO
    n_corte = slice
    salida = img_data
    masc_aux1 = masc_aux2 = img_dilation
    #SEGMENTA DESDE EL SLICE HASTA EL FINAL
    while n_corte < img_data.shape[2]:
        nuevo_corte = np.array(salida[:,:,a-n_corte])
        nuevo_corte = nuevo_corte*masc_aux1
        for i in range(nuevo_corte.shape[0]):
            for j in range(nuevo_corte.shape[1]):
                if nuevo_corte[i,j] <= int_min:
                    salida[i,j,a-n_corte] = 0
        aux2, masc_aux1 = cv2.threshold(np.uint8(salida[:,:,a-n_corte]), 1, 255, cv2.THRESH_BINARY)
        masc_aux1 = np.array(np.float64(masc_aux1/255))
        n_corte += 1
    n_corte = slice-1
    #SEGMENTA DESDE EL SLICE HASTA EL PRINCIPIO
    while n_corte >= 0:
        nuevo_corte = np.array(salida[:,:,a-n_corte])
        nuevo_corte = nuevo_corte*masc_aux2
        for i in range(nuevo_corte.shape[0]):
            for j in range(nuevo_corte.shape[1]):
                if nuevo_corte[i,j] <= int_min:
                    salida[i,j,a-n_corte] = 0
        aux3, masc_aux2 = cv2.threshold(np.uint8(salida[:,:,a-n_corte]), 1, 255, cv2.THRESH_BINARY)
        masc_aux2 = np.array(np.float64(masc_aux2/255))
        n_corte = n_corte - 1
        
    #GUARDA EL TUMOR ENUN ARCHIVO .nii
    #tumor_completo = nib.nifti1.Nifti1Image(salida, None, header=img.header)
    #nib.save(tumor_completo, 'Tumor_'+imagen_original_nii)

    #CONTADORDE VOXEL
    cont = 0
    cont_voxel = 0
    int_voxel = []
    while cont < salida.shape[2]:
        for i in range(salida.shape[0]):
            for j in range(salida.shape[1]):
                if salida[i,j,cont] != 0:
                    int_voxel.append(salida[i,j,cont])
                    cont_voxel += 1
        cont += 1
    cv2.imwrite('Tumor_delineado_'+imagen_original_nii+'.jpg', bordet)
    return corte, tumor, cont_voxel, img.header.get_zooms(), borde, bordet, int_voxel

# CASOS FUNCIONALES
# CASO 1
corte1, tumor1, n_voxel1, header1, borde1, borde_tumor1, int_voxel1= tumor_extraction('tumores/case_001_2.nii', 6)
vol_tumor1 = n_voxel1*(header1[0]*header1[1]*header1[2])
print('Volumen Tumor caso 1: ' + str(vol_tumor1) + 'mm^3')
# CASO 7
corte7, tumor7, n_voxel7, header7, borde7, borde_tumor7, int_voxel7 = tumor_extraction('tumores/case_007_2.nii', 20)
vol_tumor7 = n_voxel7*(header7[0]*header7[1]*header7[2])
print('Volumen Tumor caso 7: ' + str(vol_tumor7) + 'mm^3')
# CASO 14
corte14, tumor14, n_voxel14, header14, borde14, borde_tumor14, int_voxel14 = tumor_extraction('tumores/case_014_2.nii', 17)
vol_tumor14 = n_voxel14*(header14[0]*header14[1]*header14[2])
print('Volumen Tumor caso 14: ' + str(vol_tumor14) + 'mm^3')
#CASO 58
corte58, tumor58, n_voxel58, header58, borde58, borde_tumor58, int_voxel58 = tumor_extraction('tumores/case_058_2.nii', 31)
vol_tumor58 = n_voxel58*(header58[0]*header58[1]*header58[2])
print('Volumen Tumor caso 58: ' + str(vol_tumor58) + 'mm^3')

# CASOS NO FUNCIONALES
#corte4, tumor4, n_voxel4, header4, borde4, borde_tumor4, int_voxel4 = tumor_extraction('case_004_2.nii', 72)
#corte5, tumor5, n_voxel5, header5, borde5, borde_tumor5, int_voxel5 = tumor_extraction('case_005_2.nii', 59)
#corte8, tumor8, n_voxel8, header8, borde8, borde_tumor8, int_voxel8 = tumor_extraction('case_008_2.nii', 11)

# GRAFICAS

# CASO 1
plt.figure(1, figsize=(10,10))
plt.subplot(221), plt.imshow(corte1, cmap=plt.cm.gist_gray), plt.title('Corte')
plt.subplot(222), plt.imshow(tumor1, cmap=plt.cm.gist_gray), plt.title('Tumor segmentado')
plt.subplot(223), plt.imshow(borde1, cmap=plt.cm.gist_yarg), plt.title('Borde')
plt.subplot(224), plt.imshow(borde_tumor1, cmap=plt.cm.gist_yarg), plt.title('Borde Tumor')
plt.tight_layout(pad =6, w_pad=2,h_pad=2)
plt.show()
# CASO 7
plt.figure(2, figsize=(10,10))
plt.subplot(221), plt.imshow(corte7, cmap=plt.cm.gist_gray), plt.title('Corte')
plt.subplot(222), plt.imshow(tumor7, cmap=plt.cm.gist_gray), plt.title('Tumor segmentado')
plt.subplot(223), plt.imshow(borde7, cmap=plt.cm.gist_yarg), plt.title('Borde')
plt.subplot(224), plt.imshow(borde_tumor7, cmap=plt.cm.gist_yarg), plt.title('Borde Tumor')
plt.tight_layout(pad =6, w_pad=2,h_pad=2)
plt.show()
# CASO 14
plt.figure(3, figsize=(10,10))
plt.subplot(221), plt.imshow(corte14, cmap=plt.cm.gist_gray), plt.title('Corte')
plt.subplot(222), plt.imshow(tumor14, cmap=plt.cm.gist_gray), plt.title('Tumor segmentado')
plt.subplot(223), plt.imshow(borde14, cmap=plt.cm.gist_yarg), plt.title('Borde')
plt.subplot(224), plt.imshow(borde_tumor14, cmap=plt.cm.gist_yarg), plt.title('Borde Tumor')
plt.tight_layout(pad =6, w_pad=2,h_pad=2)
plt.show()
# CASO 58
plt.figure(4, figsize=(10,10))
plt.subplot(221), plt.imshow(corte58, cmap=plt.cm.gist_gray), plt.title('Corte')
plt.subplot(222), plt.imshow(tumor58, cmap=plt.cm.gist_gray), plt.title('Tumor segmentado')
plt.subplot(223), plt.imshow(borde58, cmap=plt.cm.gist_yarg), plt.title('Borde')
plt.subplot(224), plt.imshow(borde_tumor58, cmap=plt.cm.gist_yarg), plt.title('Borde Tumor')
plt.tight_layout(pad =6, w_pad=2,h_pad=2)
plt.show()

'''
# GUARDAR .png CASO 14
#tumor14 = tumor14.astype(np.uint8)
#tum = Image.fromarray(tumor14)
#tum.save("tumor_segmentado.png")
#borde_tumor14 = borde_tumor14.astype(np.uint8)
#bord = Image.fromarray(borde_tumor14)
#bord.save("borde_tumor.png")
'''

cv2.waitKey(0)
cv2.destroyAllWindows()