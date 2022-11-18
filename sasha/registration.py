"""
Registration module for masks and higher level analysis

"""

import os
import numpy as np
from dipy.viz import regtools
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


def register_3d(static, moving, static_affine, moving_affine, dof=12):
    static_grid2world = static_affine
    moving_grid2world = moving_affine
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    nbins = 64
    level_iters = [10000, 1000, 100]
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    affreg = AffineRegistration(metric=metric, level_iters=level_iters)
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)
    if dof == 3:
        transformed = translation.transform(moving)
        return transformed, translation

    if dof > 3:
        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=starting_affine)

    if dof == 6:
        transformed = rigid.transform(moving)
        return transformed, rigid

    if dof == 12:
        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        affine = affreg.optimize(static, moving, transform, params0,
                                 static_grid2world, moving_grid2world,
                                 starting_affine=starting_affine)
        transformed = affine.transform(moving)
        return transformed, affine


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



