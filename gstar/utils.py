"""Functions to make weight matrices"""
import numpy as np


def make_impact(idx, max_s_lag, mode=None, nearest=None):
    s_lag_impact = np.zeros(max_s_lag)
    for i in range(max_s_lag):
        if i <= idx:
            s_lag_impact[i] = (i - idx) % max_s_lag
        else:
            s_lag_impact[i] = max_s_lag - (i - idx) % max_s_lag
    
    if mode == 'binary':
        s_lag_impact *= (s_lag_impact == nearest)
        
    return s_lag_impact

def make_masks(dist_mat, max_s_lag, grid):
    masks = []
    masks_count = np.zeros((max_s_lag, len(dist_mat)))
    for i in range(max_s_lag):
        mask = np.logical_and(dist_mat > grid[i], dist_mat <= grid[i+1])
        masks.append(mask)
        masks_count[i,:] = mask.sum(axis=1)
    return masks, masks_count

def make_weight_matrices(dist_mat, max_s_lag, grid, mode, func=None):
    dist_mat = np.asarray(dist_mat)
    masks, masks_count = make_masks(dist_mat, max_s_lag, grid)
    d_num = len(dist_mat)
    weights = [np.eye(d_num)]
    
    # binary
    if mode == 'binary':
        for i in range(max_s_lag):
            mat = np.multiply(masks[i], dist_mat)
            closest = np.where(mat==0, np.inf, mat).min(axis=1)
            for j in range(d_num):
                mat[j] = (mat[j] == closest[j]).astype(int)
                nearest = max_s_lag - 1
                while mat[j].sum() == 0:
                    impact = make_impact(i, max_s_lag, mode, nearest)
                    denum = np.dot(masks_count[:,j], impact)
                    np.nan_to_num(mat[j], copy=False)
                    if denum != 0:
                        for k in range(max_s_lag):
                            mat[j] += masks[k][j] * impact[k] / denum    
                    nearest -= 1
                    if nearest == 0: break
            assert(np.allclose(np.diag(mat), 0))
            assert(np.allclose(mat.sum(axis=1), 1))
            weights.append(mat)
    
    # uniform
    elif mode == 'uniform':
        for i in range(max_s_lag):
            impact = make_impact(i, max_s_lag)
            mat = masks[i].astype(float)
            for j in range(d_num):
                if masks_count[i,j] == 0:
                    denum = np.dot(masks_count[:,j], impact)
                    for k in range(max_s_lag):
                        mat[j] += masks[k][j] * impact[k] / denum
                else:
                    mat[j] /= masks_count[i,j]
            assert(np.allclose(np.diag(mat), 0))
            assert(np.allclose(mat.sum(axis=1), 1))
            weights.append(mat)

    # non-uniform
    elif mode == 'non-uniform' and func is not None:
        mats = []
        for i in range(max_s_lag):
            mat = np.multiply(masks[i], dist_mat)
            mat = func(mat)
            mat[mat == 1] = 0
            mats.append(mat)
        
        for i in range(max_s_lag):
            impact = make_impact(i, max_s_lag)
            mat = mats[i].copy()
            for j in range(d_num):
                if mat[j].sum() == 0:
                    mat[j] = 0
                    for k in range(max_s_lag):
                        mat[j] += np.nan_to_num(mats[k][j]) * impact[k]
                mat[j] /= mat[j].sum()
            assert(np.allclose(np.diag(mat), 0))
            assert(np.allclose(mat.sum(axis=1), 1))
            weights.append(mat)
    
    # others
    elif func is None:
        raise ValueError("Must input underlying function for 'non-uniform' mode.")
    else:
        raise ValueError("Mode not recognized. Try one of 'binary', 'uniform', or 'non-uniform'.")
        
    return weights