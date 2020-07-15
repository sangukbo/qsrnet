import numpy as np

def get_object_ids(object_names, class_names):
    object_ids = []
    for object_name in object_names:
        object_ids.append(class_names.index(object_name))
    return object_ids

def get_pair_ids(pair_names, class_names):
    pair_ids = []
    for pair_name in pair_names:
        obj1_name, obj2_name = pair_name[1:-1].split(',')
        obj1_id = class_names.index(obj1_name); obj2_id = class_names.index(obj2_name)
        pair_ids.append('(' + str(obj1_id) + ',' + str(obj2_id) + ')')
    return pair_ids

def pair_name_from_id(paid_id, class_names):
    obj1_id, obj2_id = pair_id[1:-1].split(',')
    obj1_name = class_names[obj1_id]; obj2_name = class_names[obj2_id]
    return '(' + obj1_name + ',' + obj2_name + ')'

def masks_for_objects_of_interest(mask_results, object_ids):
    # compute new mask_results for objects in object_ids
    final_size = 0
    for i in range(len(mask_results['class_ids'])):
        if mask_results['class_ids'][i] in object_ids:
            final_size = final_size + 1
    new_class_ids = np.empty(final_size, dtype = int)
    new_masks = np.zeros((final_size, mask_results['masks'].shape[1], mask_results['masks'].shape[2]), dtype = bool)
    new_rois = np.zeros((final_size, 4), dtype = int)
    id_iteration = 0
    for i in range(len(mask_results['class_ids'])):
        if mask_results['class_ids'][i] in object_ids:
            new_class_ids[id_iteration] = mask_results['class_ids'][i]
            new_masks[id_iteration, :, :] = mask_results['masks'][i, :, :]
            new_rois[id_iteration, :] = mask_results['rois'][i, :]
            id_iteration = id_iteration + 1
    return {'class_ids': new_class_ids, 'masks': new_masks, 'rois': new_rois}
