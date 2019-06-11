from __future__ import print_function

import SimpleITK as sitk
import sys
import os

if len ( sys.argv ) < 2:
    print( "Usage: N4BiasFieldCorrection inputImage " + \
        "outputImage [shrinkFactor] [maskImage] [numberOfIterations] " +\
        "[numberOfFittingLevels]" )
    sys.exit ( 1 )


inputImage = sitk.ReadImage( sys.argv[1] )

if len ( sys.argv ) > 4:
    maskImage = sitk.ReadImage( sys.argv[4], sitk.sitkUint8 )

edge_images = []
comp_images = []
mask_images = []
for i in range(inputImage.GetNumberOfComponentsPerPixel()):
    component_image = sitk.VectorIndexSelectionCast(inputImage, i, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold(component_image, 0, 1, 200 )
    comp_images.append(component_image)
    mask_images.append(maskImage)

zipped = zip(comp_images, mask_images)
# if len ( sys.argv ) > 3:
#     inputImage = sitk.Shrink( inputImage, [ int(sys.argv[3]) ] * inputImage.GetDimension() )
#     maskImage = sitk.Shrink( maskImage, [ int(sys.argv[3]) ] * inputImage.GetDimension() )

# inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )

corrector = sitk.N4BiasFieldCorrectionImageFilter();

numberFittingLevels = 3
outputs = []
for component, mask in zipped:
    output = corrector.Execute( component, mask )
    outputs.append(output)
    
edge_image = sitk.Compose(outputs)
print(edge_image)
sitk.WriteImage(edge_image, sys.argv[2])
# sitk.Show(edge_image, "N4 Corrected", debugOn=True)



