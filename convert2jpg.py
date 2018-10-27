# -*- coding=utf-8 -*-
import os
from glob import glob
from datasets.medicalImage import read_mhd_image, fill_region
from config import type2pixel, pixel2type
import numpy as np
from xml.dom.minidom import Document
import cv2

image_suffix_name = 'jpg'

def LiverLesionDetection_Iterator(image_dir, execute_func, *parameters):
    '''
    遍历MICCAI2018文件夹的框架
    :param execute_func:
    :return:
    '''
    for sub_name in ['train', 'val', 'test']:
        names = os.listdir(os.path.join(image_dir, sub_name))
        for name in names:
            cur_slice_dir = os.path.join(image_dir, sub_name, name)
            execute_func(cur_slice_dir, *parameters)

def extract_bboxs_mask_from_mask(mask_image, tumor_types):
    mask_image = mask_image[1, :, :]
    w, h = np.shape(mask_image)
    if w != 512 or h != 512:
        print(np.shape(mask_image))
        assert False
    with open(tumor_types, 'r') as f:
        lines = f.readlines()
        idx2names = {}
        for line in lines:
            line = line[:-1]
            idx, name = line.split(' ')
            idx2names[idx] = name

        maximum = np.max(mask_image)
        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        names = []
        res_mask = np.zeros_like(mask_image)
        for i in range(1, maximum + 1):
            cur_mask_image = np.asarray(mask_image == i, np.uint8)
            if np.sum(cur_mask_image) == 0:
                continue
            filled_mask = fill_region(cur_mask_image)
            filled_mask[filled_mask == 1] = type2pixel[idx2names[str(i)]][1]
            res_mask[filled_mask != 0] = filled_mask[filled_mask != 0]
            xs, ys = np.where(cur_mask_image == 1)
            min_x = np.min(xs)
            min_y = np.min(ys)
            max_x = np.max(xs)
            max_y = np.max(ys)
            min_xs.append(min_x)
            min_ys.append(min_y)
            max_xs.append(max_x)
            max_ys.append(max_y)
            names.append(idx2names[str(i)])
    return min_xs, min_ys, max_xs, max_ys, names, res_mask


def extract_bboxs_from_mask(mask_image, tumor_types):
    mask_image = mask_image[1, :, :]
    w, h = np.shape(mask_image)
    if w != 512 or h != 512:
        print(np.shape(mask_image))
        assert False
    with open(tumor_types, 'r') as f:
        lines = f.readlines()
        idx2names = {}
        for line in lines:
            line = line[:-1]
            idx, name = line.split(' ')
            idx2names[idx] = name

        maximum = np.max(mask_image)
        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        names = []
        for i in range(1, maximum + 1):
            cur_mask_image = np.asarray(mask_image == i, np.uint8)
            if np.sum(cur_mask_image) == 0:
                continue
            xs, ys = np.where(cur_mask_image == 1)
            min_x = np.min(xs)
            min_y = np.min(ys)
            max_x = np.max(xs)
            max_y = np.max(ys)
            min_xs.append(min_x)
            min_ys.append(min_y)
            max_xs.append(max_x)
            max_ys.append(max_y)
            names.append(idx2names[str(i)])
    return min_xs, min_ys, max_xs, max_ys, names

def dicom2jpg_singlephase(slice_dir, save_dir, phase_name='PV'):
    '''
    前置条件：已经将dicom格式的数据转为成MHD格式，并且已经提出了slice，一个mhd文件只包含了三个slice
    将单个phase的mhd转化为jpg格式
    :param slice_dir:
    :param save_dir:
    :param phase_name:
    :return:
    '''
    mhd_image_path = os.path.join(slice_dir, 'Image_' + phase_name + '.mhd')
    mhd_mask_path = os.path.join(slice_dir, 'Mask_' + phase_name + '.mhd')
    mhd_image = read_mhd_image(mhd_image_path)
    mask_image = read_mhd_image(mhd_mask_path)
    mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
    mhd_image = np.transpose(mhd_image, axes=[1, 2, 0])
    # mhd_image = np.expand_dims(mhd_image, axis=2)
    # mhd_image = np.concatenate([mhd_image, mhd_image, mhd_image], axis=2)

    mask_image = np.asarray(np.squeeze(mask_image), np.uint8)
    max_v = 300.
    min_v = -350.
    mhd_image[mhd_image > max_v] = max_v
    mhd_image[mhd_image < min_v] = min_v
    print(np.mean(mhd_image, dtype=np.float32))
    mhd_image -= np.mean(mhd_image)
    min_v = np.min(mhd_image)
    max_v = np.max(mhd_image)
    interv = max_v - min_v
    mhd_image = (mhd_image - min_v) / interv
    file_name = os.path.basename(slice_dir)
    dataset_name = os.path.basename(os.path.dirname(slice_dir))
    save_path = os.path.join(save_dir, phase_name, dataset_name, file_name+'.jpg')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    print('the shape of mask_image is ', np.shape(mask_image))
    _, _, depth_image= np.shape(mhd_image)
    if depth_image == 2:
        mhd_image = np.concatenate(
            [mhd_image, np.expand_dims(mhd_image[:, :, np.argmax(np.sum(np.sum(mask_image, axis=1), axis=1))], axis=2)],
            axis=2
        )
        print('Error')
    cv2.imwrite(save_path, mhd_image * 255)

    xml_save_dir = os.path.join(save_dir, phase_name, dataset_name+'_xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    evulate_gt_dir = os.path.join(save_dir, phase_name, dataset_name+'_gt')
    if not os.path.exists(evulate_gt_dir):
        os.makedirs(evulate_gt_dir)

    xml_save_path  = os.path.join(xml_save_dir, file_name + '.xml')
    gt_save_path = os.path.join(evulate_gt_dir, file_name + '.txt') # for evulate

    doc = Document()
    root_node = doc.createElement('annotation')
    doc.appendChild(root_node)

    folder_name = os.path.basename(save_dir) + '/' + phase_name
    folder_node = doc.createElement('folder')
    root_node.appendChild(folder_node)
    folder_txt_node = doc.createTextNode(folder_name)
    folder_node.appendChild(folder_txt_node)

    file_name = file_name + '.jpg'
    filename_node = doc.createElement('filename')
    root_node.appendChild(filename_node)
    filename_txt_node = doc.createTextNode(file_name)
    filename_node.appendChild(filename_txt_node)

    shape = list(np.shape(mhd_image))
    size_node = doc.createElement('size')
    root_node.appendChild(size_node)
    width_node = doc.createElement('width')
    width_node.appendChild(doc.createTextNode(str(shape[0])))
    height_node = doc.createElement('height')
    height_node.appendChild(doc.createTextNode(str(shape[1])))
    depth_node = doc.createElement('depth')
    depth_node.appendChild(doc.createTextNode(str(3)))
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)

    # mask_image[mask_image != 1] = 0
    # xs, ys = np.where(mask_image == 1)
    # min_x = np.min(xs)
    # min_y = np.min(ys)
    # max_x = np.max(xs)
    # max_y = np.max(ys)
    min_xs, min_ys, max_xs, max_ys, names = extract_bboxs_from_mask(mask_image, os.path.join(slice_dir, 'tumor_types'))
    lines = []
    for min_x, min_y, max_x, max_y, name in zip(min_xs, min_ys, max_xs, max_ys, names):
        object_node = doc.createElement('object')
        root_node.appendChild(object_node)
        name_node = doc.createElement('name')
        name_node.appendChild(doc.createTextNode(name))
        object_node.appendChild(name_node)
        truncated_node = doc.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_node.appendChild(doc.createTextNode('0'))
        difficult_node = doc.createElement('difficult')
        object_node.appendChild(difficult_node)
        difficult_node.appendChild(doc.createTextNode('0'))

        bndbox_node = doc.createElement('bndbox')
        object_node.appendChild(bndbox_node)
        xmin_node = doc.createElement('xmin')
        xmin_node.appendChild(doc.createTextNode(str(min_y)))
        bndbox_node.appendChild(xmin_node)

        ymin_node = doc.createElement('ymin')
        ymin_node.appendChild(doc.createTextNode(str(min_x)))
        bndbox_node.appendChild(ymin_node)

        xmax_node = doc.createElement('xmax')
        xmax_node.appendChild(doc.createTextNode(str(max_y)))
        bndbox_node.appendChild(xmax_node)

        ymax_node = doc.createElement('ymax')
        ymax_node.appendChild(doc.createTextNode(str(max_x)))
        bndbox_node.appendChild(ymax_node)

        line = '%s %d %d %d %d\n' % (name, min_y, min_x, max_y, max_x)
        print(line)

        lines.append(line)


    with open(xml_save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    with open(gt_save_path, 'w') as f:
        f.writelines(lines)
        f.close()


def dicom2jpg_multiphase(slice_dir, save_dir, phase_names, target_phase_name):
    '''
    前置条件：已经将dicom格式的数据转为成MHD格式，并且已经提出了slice，一个mhd文件只包含了三个slice
    针对每个phase只是用一个slice
    :param slice_dir:
    :param save_dir:
    :param phase_names:
    :param target_phase_name:
    :return:
    '''
    total_phase_name = ''.join(phase_names)
    target_phase_mask = None
    mhd_images = []
    for phase_name in phase_names:
        mhd_image_path = os.path.join(slice_dir, 'Image_' + phase_name + '.mhd')
        mhd_mask_path = os.path.join(slice_dir, 'Mask_' + phase_name + '.mhd')
        mhd_image = read_mhd_image(mhd_image_path)
        mask_image = read_mhd_image(mhd_mask_path)
        mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
        mhd_image = np.transpose(mhd_image, axes=[1, 2, 0])
        if phase_name == target_phase_name:
            target_phase_mask = mask_image
        mhd_images.append(mhd_image)
    # mhd_image = np.expand_dims(mhd_image, axis=2)
    # mhd_image = np.concatenate([mhd_image, mhd_image, mhd_image], axis=2)
    mhd_image = np.concatenate([np.expand_dims(ele[:, :, 1], axis=2) for ele in mhd_images], axis=-1)
    mask_image = target_phase_mask
    mask_image = np.asarray(np.squeeze(mask_image), np.uint8)
    max_v = 300.
    min_v = -350.
    mhd_image[mhd_image > max_v] = max_v
    mhd_image[mhd_image < min_v] = min_v
    print(np.mean(mhd_image, dtype=np.float32))
    mhd_image -= np.mean(mhd_image)
    min_v = np.min(mhd_image)
    max_v = np.max(mhd_image)
    interv = max_v - min_v
    mhd_image = (mhd_image - min_v) / interv
    file_name = os.path.basename(slice_dir)
    dataset_name = os.path.basename(os.path.dirname(slice_dir))
    save_path = os.path.join(save_dir, total_phase_name, dataset_name, file_name+'.jpg')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    print('the shape of mask_image is ', np.shape(mask_image))
    _, _, depth_image= np.shape(mhd_image)
    if depth_image == 2:
        mhd_image = np.concatenate(
            [mhd_image, np.expand_dims(mhd_image[:, :, np.argmax(np.sum(np.sum(mask_image, axis=1), axis=1))], axis=2)],
            axis=2
        )
        print('Error')
    #mhd_image = np.asarray(mhd_image * 255, np.uint8)
    #mhd_image.tofile(save_path)
    # np.save(save_path, mhd_image * 255)
    cv2.imwrite(save_path, mhd_image * 255)

    xml_save_dir = os.path.join(save_dir, total_phase_name, dataset_name+'_xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    evulate_gt_dir = os.path.join(save_dir, total_phase_name, dataset_name+'_gt')
    if not os.path.exists(evulate_gt_dir):
        os.makedirs(evulate_gt_dir)

    xml_save_path  = os.path.join(xml_save_dir, file_name + '.xml')
    gt_save_path = os.path.join(evulate_gt_dir, file_name + '.txt') # for evulate

    doc = Document()
    root_node = doc.createElement('annotation')
    doc.appendChild(root_node)

    folder_name = os.path.basename(save_dir) + '/' + total_phase_name
    folder_node = doc.createElement('folder')
    root_node.appendChild(folder_node)
    folder_txt_node = doc.createTextNode(folder_name)
    folder_node.appendChild(folder_txt_node)

    file_name = file_name + '.jpg'
    filename_node = doc.createElement('filename')
    root_node.appendChild(filename_node)
    filename_txt_node = doc.createTextNode(file_name)
    filename_node.appendChild(filename_txt_node)

    shape = list(np.shape(mhd_image))
    size_node = doc.createElement('size')
    root_node.appendChild(size_node)
    width_node = doc.createElement('width')
    width_node.appendChild(doc.createTextNode(str(shape[0])))
    height_node = doc.createElement('height')
    height_node.appendChild(doc.createTextNode(str(shape[1])))
    depth_node = doc.createElement('depth')
    depth_node.appendChild(doc.createTextNode(str(3)))
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)

    # mask_image[mask_image != 1] = 0
    # xs, ys = np.where(mask_image == 1)
    # min_x = np.min(xs)
    # min_y = np.min(ys)
    # max_x = np.max(xs)
    # max_y = np.max(ys)
    min_xs, min_ys, max_xs, max_ys, names = extract_bboxs_from_mask(mask_image, os.path.join(slice_dir, 'tumor_types'))
    lines = []
    for min_x, min_y, max_x, max_y, name in zip(min_xs, min_ys, max_xs, max_ys, names):
        object_node = doc.createElement('object')
        root_node.appendChild(object_node)
        name_node = doc.createElement('name')
        name_node.appendChild(doc.createTextNode(name))
        object_node.appendChild(name_node)
        truncated_node = doc.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_node.appendChild(doc.createTextNode('0'))
        difficult_node = doc.createElement('difficult')
        object_node.appendChild(difficult_node)
        difficult_node.appendChild(doc.createTextNode('0'))

        bndbox_node = doc.createElement('bndbox')
        object_node.appendChild(bndbox_node)
        xmin_node = doc.createElement('xmin')
        xmin_node.appendChild(doc.createTextNode(str(min_y)))
        bndbox_node.appendChild(xmin_node)

        ymin_node = doc.createElement('ymin')
        ymin_node.appendChild(doc.createTextNode(str(min_x)))
        bndbox_node.appendChild(ymin_node)

        xmax_node = doc.createElement('xmax')
        xmax_node.appendChild(doc.createTextNode(str(max_y)))
        bndbox_node.appendChild(xmax_node)

        ymax_node = doc.createElement('ymax')
        ymax_node.appendChild(doc.createTextNode(str(max_x)))
        bndbox_node.appendChild(ymax_node)

        line = '%s %d %d %d %d\n' % (name, min_y, min_x, max_y, max_x)
        print(line)

        lines.append(line)

    with open(xml_save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    with open(gt_save_path, 'w') as f:
        f.writelines(lines)
        f.close()

def dicom2jpg_multiphase_tripleslice(slice_dir, save_dir, phase_names, target_phase_name):
    '''
    前置条件：已经将dicom格式的数据转为成MHD格式，并且已经提出了slice，一个mhd文件只包含了三个slice
    针对每个phase提取三个slice(all), 但是mask还是只提取一个
    保存的格式是name_nc.jpg, name_art.jpg, name_pv.jpg
    :param slice_dir:
    :param save_dir:
    :param phase_names:
    :param target_phase_name:
    :return:
    '''
    total_phase_name = ''.join(phase_names)
    total_phase_name += '_tripleslice'
    target_phase_mask = None
    mhd_images = []
    for phase_name in phase_names:
        mhd_image_path = os.path.join(slice_dir, 'Image_' + phase_name + '.mhd')
        mhd_mask_path = os.path.join(slice_dir, 'Mask_' + phase_name + '.mhd')
        mhd_image = read_mhd_image(mhd_image_path)
        mask_image = read_mhd_image(mhd_mask_path)
        mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
        mhd_image = np.transpose(mhd_image, axes=[1, 2, 0])
        if phase_name == target_phase_name:
            target_phase_mask = mask_image
        _, _, depth_image = np.shape(mhd_image)
        if depth_image == 2:
            mhd_image = np.concatenate(
                [mhd_image,
                 np.expand_dims(mhd_image[:, :, np.argmax(np.sum(np.sum(mask_image, axis=1), axis=1))], axis=2)],
                axis=2
            )
            print('Error')
        mhd_images.append(mhd_image)
    # mhd_image = np.expand_dims(mhd_image, axis=2)
    # mhd_image = np.concatenate([mhd_image, mhd_image, mhd_image], axis=2)
    mhd_image = np.concatenate(mhd_images, axis=-1)
    mask_image = target_phase_mask
    mask_image = np.asarray(np.squeeze(mask_image), np.uint8)
    max_v = 300.
    min_v = -350.
    mhd_image[mhd_image > max_v] = max_v
    mhd_image[mhd_image < min_v] = min_v
    print(np.mean(mhd_image, dtype=np.float32))
    mhd_image -= np.mean(mhd_image)
    min_v = np.min(mhd_image)
    max_v = np.max(mhd_image)
    interv = max_v - min_v
    mhd_image = (mhd_image - min_v) / interv
    file_name = os.path.basename(slice_dir)
    dataset_name = os.path.basename(os.path.dirname(slice_dir))


    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    print('the shape of mask_image is ', np.shape(mask_image))
    for phase_idx, phase_name in enumerate(['NNC', 'ART', 'PPV']):
        save_path = os.path.join(save_dir, total_phase_name, dataset_name, file_name + '_%s.jpg' % phase_name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, mhd_image[:, :, phase_idx * 3: (phase_idx + 1) * 3] * 255)

    xml_save_dir = os.path.join(save_dir, total_phase_name, dataset_name+'_xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    evulate_gt_dir = os.path.join(save_dir, total_phase_name, dataset_name+'_gt')
    if not os.path.exists(evulate_gt_dir):
        os.makedirs(evulate_gt_dir)

    xml_save_path  = os.path.join(xml_save_dir, file_name + '.xml')
    gt_save_path = os.path.join(evulate_gt_dir, file_name + '.txt') # for evulate

    doc = Document()
    root_node = doc.createElement('annotation')
    doc.appendChild(root_node)

    folder_name = os.path.basename(save_dir) + '/' + total_phase_name
    folder_node = doc.createElement('folder')
    root_node.appendChild(folder_node)
    folder_txt_node = doc.createTextNode(folder_name)
    folder_node.appendChild(folder_txt_node)

    file_name = file_name + '.jpg'
    filename_node = doc.createElement('filename')
    root_node.appendChild(filename_node)
    filename_txt_node = doc.createTextNode(file_name)
    filename_node.appendChild(filename_txt_node)

    shape = list(np.shape(mhd_image))
    size_node = doc.createElement('size')
    root_node.appendChild(size_node)
    width_node = doc.createElement('width')
    width_node.appendChild(doc.createTextNode(str(shape[0])))
    height_node = doc.createElement('height')
    height_node.appendChild(doc.createTextNode(str(shape[1])))
    depth_node = doc.createElement('depth')
    depth_node.appendChild(doc.createTextNode(str(3)))
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)

    # mask_image[mask_image != 1] = 0
    # xs, ys = np.where(mask_image == 1)
    # min_x = np.min(xs)
    # min_y = np.min(ys)
    # max_x = np.max(xs)
    # max_y = np.max(ys)
    min_xs, min_ys, max_xs, max_ys, names = extract_bboxs_from_mask(mask_image, os.path.join(slice_dir, 'tumor_types'))
    lines = []
    for min_x, min_y, max_x, max_y, name in zip(min_xs, min_ys, max_xs, max_ys, names):
        object_node = doc.createElement('object')
        root_node.appendChild(object_node)
        name_node = doc.createElement('name')
        name_node.appendChild(doc.createTextNode(name))
        object_node.appendChild(name_node)
        truncated_node = doc.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_node.appendChild(doc.createTextNode('0'))
        difficult_node = doc.createElement('difficult')
        object_node.appendChild(difficult_node)
        difficult_node.appendChild(doc.createTextNode('0'))

        bndbox_node = doc.createElement('bndbox')
        object_node.appendChild(bndbox_node)
        xmin_node = doc.createElement('xmin')
        xmin_node.appendChild(doc.createTextNode(str(min_y)))
        bndbox_node.appendChild(xmin_node)

        ymin_node = doc.createElement('ymin')
        ymin_node.appendChild(doc.createTextNode(str(min_x)))
        bndbox_node.appendChild(ymin_node)

        xmax_node = doc.createElement('xmax')
        xmax_node.appendChild(doc.createTextNode(str(max_y)))
        bndbox_node.appendChild(xmax_node)

        ymax_node = doc.createElement('ymax')
        ymax_node.appendChild(doc.createTextNode(str(max_x)))
        bndbox_node.appendChild(ymax_node)

        line = '%s %d %d %d %d\n' % (name, min_y, min_x, max_y, max_x)
        print(line)

        lines.append(line)

    with open(xml_save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    with open(gt_save_path, 'w') as f:
        f.writelines(lines)
        f.close()

def dicom2jpg_multiphase_tripleslice_mask(slice_dir, save_dir, phase_names, target_phase_name):
    '''
    前置条件：已经将dicom格式的数据转为成MHD格式，并且已经提出了slice，一个mhd文件只包含了三个slice
    针对每个phase提取三个slice(all), 但是mask还是只提取一个
    保存的格式是name_nc.jpg, name_art.jpg, name_pv.jpg
    :param slice_dir:
    :param save_dir:
    :param phase_names:
    :param target_phase_name:
    :return:
    '''
    total_phase_name = ''.join(phase_names)
    total_phase_name += '_tripleslice_mask'
    target_phase_mask = None
    mhd_images = []
    for phase_name in phase_names:
        mhd_image_path = os.path.join(slice_dir, 'Image_' + phase_name + '.mhd')
        mhd_mask_path = os.path.join(slice_dir, 'Mask_' + phase_name + '.mhd')
        mhd_image = read_mhd_image(mhd_image_path)
        mask_image = read_mhd_image(mhd_mask_path)
        mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
        mhd_image = np.transpose(mhd_image, axes=[1, 2, 0])
        if phase_name == target_phase_name:
            target_phase_mask = mask_image
        _, _, depth_image = np.shape(mhd_image)
        if depth_image == 2:
            mhd_image = np.concatenate(
                [mhd_image,
                 np.expand_dims(mhd_image[:, :, np.argmax(np.sum(np.sum(mask_image, axis=1), axis=1))], axis=2)],
                axis=2
            )
            print('Error')
        mhd_images.append(mhd_image)
    # mhd_image = np.expand_dims(mhd_image, axis=2)
    # mhd_image = np.concatenate([mhd_image, mhd_image, mhd_image], axis=2)
    mhd_image = np.concatenate(mhd_images, axis=-1)
    mask_image = target_phase_mask
    mask_image = np.asarray(np.squeeze(mask_image), np.uint8)
    max_v = 300.
    min_v = -350.
    mhd_image[mhd_image > max_v] = max_v
    mhd_image[mhd_image < min_v] = min_v
    print(np.mean(mhd_image, dtype=np.float32))
    mhd_image -= np.mean(mhd_image)
    min_v = np.min(mhd_image)
    max_v = np.max(mhd_image)
    interv = max_v - min_v
    mhd_image = (mhd_image - min_v) / interv
    file_name = os.path.basename(slice_dir)
    dataset_name = os.path.basename(os.path.dirname(slice_dir))


    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    print('the shape of mask_image is ', np.shape(mask_image))
    for phase_idx, phase_name in enumerate(['NNC', 'ART', 'PPV']):
        save_path = os.path.join(save_dir, total_phase_name, dataset_name,
                                 file_name + '_%s.%s' % (phase_name, image_suffix_name))
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if image_suffix_name == 'jpg':
            cv2.imwrite(save_path, mhd_image[:, :, phase_idx * 3: (phase_idx + 1) * 3] * 255)
        else:
            cv2.imwrite(save_path, np.asarray(mhd_image[:, :, phase_idx * 3: (phase_idx + 1) * 3] * 255, np.int32))
    if not os.path.exists(os.path.join(save_dir, total_phase_name, dataset_name + '_mask')):
        os.mkdir(os.path.join(save_dir, total_phase_name, dataset_name + '_mask'))
    if not os.path.exists(os.path.join(save_dir, total_phase_name, dataset_name + '_mask_vis')):
        os.mkdir(os.path.join(save_dir, total_phase_name, dataset_name + '_mask_vis'))
    real_mask_save_path = os.path.join(save_dir, total_phase_name, dataset_name + '_mask', file_name + '.' + image_suffix_name)
    vis_mask_save_path = os.path.join(save_dir, total_phase_name, dataset_name + '_mask_vis', file_name + '.' + image_suffix_name)
    xml_save_dir = os.path.join(save_dir, total_phase_name, dataset_name+'_xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    evulate_gt_dir = os.path.join(save_dir, total_phase_name, dataset_name+'_gt')
    if not os.path.exists(evulate_gt_dir):
        os.makedirs(evulate_gt_dir)

    xml_save_path  = os.path.join(xml_save_dir, file_name + '.xml')
    gt_save_path = os.path.join(evulate_gt_dir, file_name + '.txt') # for evulate

    doc = Document()
    root_node = doc.createElement('annotation')
    doc.appendChild(root_node)

    folder_name = os.path.basename(save_dir) + '/' + total_phase_name
    folder_node = doc.createElement('folder')
    root_node.appendChild(folder_node)
    folder_txt_node = doc.createTextNode(folder_name)
    folder_node.appendChild(folder_txt_node)

    file_name = file_name + '.jpg'
    filename_node = doc.createElement('filename')
    root_node.appendChild(filename_node)
    filename_txt_node = doc.createTextNode(file_name)
    filename_node.appendChild(filename_txt_node)

    shape = list(np.shape(mhd_image))
    size_node = doc.createElement('size')
    root_node.appendChild(size_node)
    width_node = doc.createElement('width')
    width_node.appendChild(doc.createTextNode(str(shape[0])))
    height_node = doc.createElement('height')
    height_node.appendChild(doc.createTextNode(str(shape[1])))
    depth_node = doc.createElement('depth')
    depth_node.appendChild(doc.createTextNode(str(3)))
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)

    # mask_image[mask_image != 1] = 0
    # xs, ys = np.where(mask_image == 1)
    # min_x = np.min(xs)
    # min_y = np.min(ys)
    # max_x = np.max(xs)
    # max_y = np.max(ys)
    min_xs, min_ys, max_xs, max_ys, names, mask = extract_bboxs_mask_from_mask(mask_image,
                                                                               os.path.join(slice_dir, 'tumor_types'))

    cv2.imwrite(vis_mask_save_path, np.asarray(mask * 50, np.int32))
    for key in pixel2type.keys():
        mask[mask == key] = type2pixel[pixel2type[key]][0]
    cv2.imwrite(real_mask_save_path, np.asarray(mask, np.int32))

    lines = []
    for min_x, min_y, max_x, max_y, name in zip(min_xs, min_ys, max_xs, max_ys, names):
        object_node = doc.createElement('object')
        root_node.appendChild(object_node)
        name_node = doc.createElement('name')
        name_node.appendChild(doc.createTextNode(name))
        object_node.appendChild(name_node)
        truncated_node = doc.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_node.appendChild(doc.createTextNode('0'))
        difficult_node = doc.createElement('difficult')
        object_node.appendChild(difficult_node)
        difficult_node.appendChild(doc.createTextNode('0'))

        bndbox_node = doc.createElement('bndbox')
        object_node.appendChild(bndbox_node)
        xmin_node = doc.createElement('xmin')
        xmin_node.appendChild(doc.createTextNode(str(min_y)))
        bndbox_node.appendChild(xmin_node)

        ymin_node = doc.createElement('ymin')
        ymin_node.appendChild(doc.createTextNode(str(min_x)))
        bndbox_node.appendChild(ymin_node)

        xmax_node = doc.createElement('xmax')
        xmax_node.appendChild(doc.createTextNode(str(max_y)))
        bndbox_node.appendChild(xmax_node)

        ymax_node = doc.createElement('ymax')
        ymax_node.appendChild(doc.createTextNode(str(max_x)))
        bndbox_node.appendChild(ymax_node)

        line = '%s %d %d %d %d\n' % (name, min_y, min_x, max_y, max_x)
        print(line)

        lines.append(line)

    with open(xml_save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    with open(gt_save_path, 'w') as f:
        f.writelines(lines)
        f.close()


if __name__ == '__main__':
    #　针对单个期相
    # image_dir = '/home/give/Documents/dataset/LiverLesionDetection_Splited/0'
    # LiverLesionDetection_Iterator(
    #     image_dir,
    #     dicom2jpg_singlephase,
    #     '/home/give/Documents/dataset/LiverLesionDetection_Splited/JPG/0',
    #     'ART'
    # )
    global image_suffix_name
    image_suffix_name = 'PNG'
    # 针对多个期相
    image_dir = '/home/give/Documents/dataset/LiverLesionDetection_Splited/0'
    LiverLesionDetection_Iterator(
        image_dir,
        dicom2jpg_multiphase_tripleslice_mask,
        '/home/give/Documents/dataset/LiverLesionDetection_Splited/JPG/0',
        ['NC', 'ART', 'PV'],
        'PV'
    )
