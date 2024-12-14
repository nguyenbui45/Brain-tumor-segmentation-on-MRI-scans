import numpy as np

def map_color(image):
    
    # check if the input is normalized or not
    if(np.all(image<=1) == True):
        image = (image*255).astype(np.uint8)

    unique_value = np.unique(image).astype(np.uint8)
    output_image = np.zeros((image.shape[0],image.shape[1],3))
    
    # generate color map based on the unique int value
    color_map={}
    color = np.array([0,0,0])
    index_switch = np.zeros(3,dtype=bool)
    index_switch[0] = True
    for element in unique_value:
        # assign a unique color to a 3 RGB value
        color_map[element] = color.copy()
        
        # for each new unique value we add 50 to the the channel that has True flag at, the flag will toggle continously. For example:
        #unique_value[0]:
        #   index_switch = [True False False]
        #   addition_color = [c1_update=c1 + 50, c2, c3]
        #unique_value[1]:
        #   modify index_switch =[False True False]
        #   addition_color = [c1_update, c2_update=c2+50, c3]
        
        # if True flag reach the last poisistion of index_switch, the next True toggle value will be roll back to the first position.
        
        # by this we only limited the update to 12 unique values. If the number of class larger than 12 than we need to either reduce the color update from 50 to smaller value or change the way to update the color
        
        idx = np.where(index_switch == True)[0][0] # index to add intensity at
        color[idx] += 50
        if idx == 2:
            index_switch[0] = True 
        else:
            index_switch[idx+1] = True
        index_switch[idx] = False
    
        
    # map 1 unique value in input_image to 3 RGB channel
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i,j,:] = color_map[image[i,j]]
            
    
    return output_image
        