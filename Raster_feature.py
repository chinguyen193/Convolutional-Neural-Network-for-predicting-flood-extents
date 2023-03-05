
#%% Function to calculate slope and curlvature

import scipy.ndimage

def feature_fast_mean(array_all, group_size, cell_size):
    '''
    calculate the slop, curvature and aspect of a terrain.
    faster implementation, use n-dimensional array calculation instead of pixel-wised calculation.
    -----
    
    about group_size:
    
    the DEM features of one pixel are calculated based on its 8 neighbors
    
    assume we merge multiple pixels into one bigger pixel, and do the same calculation,
    we get the result with the same meaning but in larger scale.
    
    the group_size indicates how many pixels are merged into one big pixel.
    or, the size of the pixel group = group_size x group_size.
    
    the mean value of each pixel group is obtained by scipy.ndimage.filters.uniform_filter
    
    the no-data pixels are filled by dilation
    
    to locate the center pixel faster, we define that the group_size must be an odd number.
    '''
    array_pad=np.pad(array_all,group_size,'edge')
    
    # fill the no-data area by dilation
    dilate_size = group_size * 3
    dilate_array = scipy.ndimage.grey_dilation(array_pad,size=(dilate_size,dilate_size))
    
    # override have-data area with the original array
    # the effect looks like offet the original array around its boundary
    indice=array_pad>0
    dilate_array[indice]=array_pad[indice]
    
    # apply mean filter
    dilate_array = scipy.ndimage.filters.uniform_filter(dilate_array,size=(group_size,group_size))
    
    group_size2=group_size+group_size
    height, width = array_all.shape

    # first row
    a=dilate_array[0:height,0:width]
    b=dilate_array[0:height,group_size:group_size+width]
    c=dilate_array[0:height,group_size2:group_size2+width]
    
    # second row
    d=dilate_array[group_size:group_size+height,0:width]
    e=dilate_array[group_size:group_size+height,group_size:group_size+width]
    f=dilate_array[group_size:group_size+height,group_size2:group_size2+width]
    
    # third row
    g=dilate_array[group_size2:group_size2+height,0:width]
    h=dilate_array[group_size2:group_size2+height,group_size:group_size+width]
    i=dilate_array[group_size2:group_size2+height,group_size2:group_size2+width]
    # as the array was dilated, no no-data should exist for pixels correspond to the pixels of e>0
    
    del(indice)
    del(array_pad)
    
    # the actual size of the pixel group (pixel num * pixel size)
    group_size = cell_size * group_size
    
    dx= ((c + 2*f + i) - (a + 2*d + g)) / (8*group_size)
    dy= ((g + 2*h + i) - (a + 2*b + c)) / (8*group_size)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) # in radian
    
    # desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm
    aspect = np.arctan2(dy, -dx)
    
    del(dx)
    del(dy)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    group_size_sq = group_size**2
    
    D = ((d + f) /2 - e) / (group_size_sq)
    E = ((b + h) /2 - e) / (group_size_sq)
    # curvature = -200 * (D + E) # why 200 here?
    
    curvature = np.clip(-group_size * (D + E),-4,4) # rescale and clip the curvature using group_size (just a trick)
    
    del(D)
    del(E)
    
    # set features of invalid pixels to 0
    indice = array_all<0
    
    cos = np.cos(aspect)
    sin = np.sin(aspect)
    
    slope[indice]=0
    cos[indice]=0
    sin[indice]=0
    curvature[indice]=0
    aspect[indice]=0

    return slope,curvature,cos,sin,aspect



