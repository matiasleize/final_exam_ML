def read_data_pantheon_plus(file_pantheon_plus,file_pantheon_plus_cov):

    '''
    Takes Pantheon+ data and extracts the data from the zhd and zhel 
    redshifts, its error dz, in addition to the data of the apparent magnitude
    with its error: mb and dm. With the errors of the apparent magnitude 
    builds the associated correlation matrix. The function returns the
    information of the redshifts, the apparent magnitude 
    and the correlation matrix inverse.
    '''

    # Read text with data

    df = pd.read_csv(file_pantheon_plus,delim_whitespace=True)
    ww = (df['zHD']>0.01) | (np.array(df['IS_CALIBRATOR'],dtype=bool))


    zhd = df['zHD'][ww]
    zhel = df['zHEL'][ww]
    mb = df['m_b_corr'][ww]

    Ccov=np.load(file_pantheon_plus_cov)['arr_0']
    Cinv=inv(Ccov)

    return zhd, zhel, Cinv, mb

# import libraries:
import sys, os
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.path.join(path_git, 'source', 'covidProcesado.csv')
ds_SN_plus = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                        'covmat_pantheon_plus_only.npz')