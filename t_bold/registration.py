"""
Registration module for masks and higher level analysis

"""

import numpy as np
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk

from dipy.viz import regtools


def nibabel_to_sitk(vol: nib.Nifti1Image) -> sitk.Image:
    data = np.transpose(np.asanyarray(vol.dataobj))
    header = vol.header
    spacing = np.array(header.get_zooms(), dtype='float64')
    affine = header.get_best_affine()
    affine[:2] *= -1
    direction = affine[:3, :3] / spacing
    origin = affine[:3, 3]
    vol_sitk = sitk.GetImageFromArray(data)
    vol_sitk.SetSpacing(spacing)
    vol_sitk.SetOrigin(origin)
    vol_sitk.SetDirection(direction.flatten())

    return vol_sitk


def sitk_to_nibabel(vol: sitk.Image) -> nib.Nifti1Image:
    vol_np = sitk.GetArrayFromImage(vol)
    vol_np = np.transpose(vol_np)
    affine = sitk_get_affine(vol)
    affine[:2] *= -1

    return nib.Nifti1Image(vol_np, affine)


def sitk_get_affine(vol: sitk.Image) -> np.array:
    origin = np.array(vol.GetOrigin())
    spacing = vol.GetSpacing()
    direction = np.reshape(vol.GetDirection(), (3, 3))
    affine = np.eye(4, 4, dtype='float64')
    affine[:3, :3] = spacing * direction
    affine[:3, 3] = origin

    return affine


def register_3d(static_fname, moving_fname):

    # load in images using nibabel, remove extra dimensions
    img_fixed = nib.load(static_fname)
    if len(img_fixed.shape) == 4:
        img_fixed = img_fixed.slice[:, :, :, 0]
    fixed = img_fixed.get_fdata()
    affine_fixed = img_fixed.affine

    img_moving = nib.load(moving_fname)
    if len(img_moving.shape) == 4:
        img_moving = img_moving.slicer[:, :, :, 0]
    moving = img_moving.get_fdata()
    affine_moving = img_moving.affine

    fixed_image = nibabel_to_sitk(img_fixed)
    moving_image = nibabel_to_sitk(img_moving)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)

    registration_method.SetMetricSamplingPercentage(0.50)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=128,
                                                                  convergenceMinimumValue=1e-8,
                                                                  convergenceWindowSize=50)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    moving_resampled_img = sitk_to_nibabel(moving_resampled)
    moving_resampled_data = moving_resampled_img.get_fdata()

    return moving_resampled_data, final_transform


def register_func2standard(data_fname, standard_fname, functional_fname, anatomical_fname, show):
    standard, standard_affine = load_nifti(standard_fname)
    functional, functional_affine = load_nifti(functional_fname)
    if len(np.shape(functional)) == 4:
        functional = functional[:, :, :, 0]
    anatomical, anatomical_affine = load_nifti(anatomical_fname)

    # functional to anatomical
    func_in_anat, affine_func2anat = register_3d(anatomical, functional,
                               anatomical_affine, functional_affine, 12)
    # anatomical to standard
    anat_in_stand, affine_anat2stand = register_3d(standard, anatomical,
                               standard_affine, anatomical_affine, 12)
    # apply transforms to data to transform to standard space
    data, data_affine = load_nifti(data_fname)
    if len(np.shape(data)) == 4:
        data = data[:, :, :, 0]
    data_anatspace = affine_func2anat.transform(data)
    data_standspace = affine_anat2stand.transform(data_anatspace)

    if show:
        func_standspace = affine_anat2stand.transform(func_in_anat)
        regtools.overlay_slices(standard, func_standspace, None, 0,
                                "Static", "Transformed")
        regtools.overlay_slices(standard, func_standspace, None, 1,
                                "Static", "Transformed")
        regtools.overlay_slices(standard, func_standspace, None, 2,
                                "Static", "Transformed")

    return data_standspace, affine_func2anat, affine_anat2stand



def register_mask2func(mask_fname, standard_fname, functional_fname, anatomical_fname, show):
    mask, mask_affine = load_nifti(mask_fname)
    standard, standard_affine = load_nifti(standard_fname)
    functional, functional_affine = load_nifti(functional_fname)
    if len(np.shape(functional)) == 4:
        functional = functional[:, :, :, 0]
    anatomical, anatomical_affine = load_nifti(anatomical_fname)

    # functional to anatomical
    func_in_anat, affine_func2anat = register_3d(anatomical, functional,
                                                 anatomical_affine, functional_affine)
    # anatomical to standard
    anat_in_stand, affine_anat2stand = register_3d(standard, anatomical,
                                                   standard_affine, anatomical_affine)



