import cv2
def undo_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width
    return do_flip_transpose(H, W, image, proposal, mask, instance, t )


def do_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    #choose one of the 8 cases

    if image    is not None: image=image.copy()
    if proposal is not None: proposal=proposal.copy()
    if mask     is not None: mask=mask.copy()
    if instance is not None: instance=instance.copy()

    if proposal is not None:
        x0 = proposal[:,1]
        y0 = proposal[:,2]
        x1 = proposal[:,3]
        y1 = proposal[:,4]

    if type==1: #rotate90
        if image is not None:
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            #mask = np.rot90(mask,k=1)
            mask = mask.transpose(1,0)
            mask = np.fliplr(mask)


        if instance is not None:
            instance = instance.transpose(1,2,0)
            #instance = np.rot90(instance,k=1)
            instance = instance.transpose(1,0,2)
            instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height-1-x0, height-1-x1

    if type==2: #rotate180
        if image is not None:
            image = cv2.flip(image,-1)

        if mask is not None:
            mask = np.rot90(mask,k=2)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.rot90(instance,k=2)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1
            y0, y1 = height-1-y0, height-1-y1

    if type==3: #rotate270
        if image is not None:
            image = image.transpose(1,0,2)
            image = cv2.flip(image,0)

        if mask is not None:
            #mask = np.rot90(mask,k=3)
            mask = mask.transpose(1,0)
            mask = np.flipud(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            #instance = np.rot90(instance,k=3)
            instance = instance.transpose(1,0,2)
            instance = np.flipud(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            y0, y1 = width-1-y0, width-1-y1

    if type==4: #flip left-right
        if image is not None:
            image = cv2.flip(image,1)

        if mask is not None:
            mask = np.fliplr(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1

    if type==5: #flip up-down
        if image is not None:
            image = cv2.flip(image,0)

        if mask is not None:
            mask = np.flipud(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.flipud(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            y0, y1 = height-1-y0, height-1-y1

    if type==6:
        if image is not None:
            image = cv2.flip(image,1)
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            mask = cv2.flip(mask,1)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.fliplr(instance)
            instance = np.rot90(instance,k=3)
            #instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height -1-x0, height -1-x1

    if type==7:
        if image is not None:
            image = cv2.flip(image,0)
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            mask = cv2.flip(mask,0)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.flipud(instance)
            instance = np.rot90(instance,k=3)
            #instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            y0, y1 = height-1-y0, height-1-y1
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height -1-x0, height -1-x1


    if proposal is not None:
        x0,x1 = np.minimum(x0,x1), np.maximum(x0,x1)
        y0,y1 = np.minimum(y0,y1), np.maximum(y0,y1)
        proposal[:,1] = x0
        proposal[:,2] = y0
        proposal[:,3] = x1
        proposal[:,4] = y1

    # if image    is not None: image=image.copy()
    # if proposal is not None: proposal=proposal.copy()
    # if mask     is not None: mask=mask.copy()
    # if instance is not None: instance=instance.copy()

    out=[]
    if image    is not None: out.append(image)
    if proposal is not None: out.append(proposal)
    if mask     is not None: out.append(mask)
    if instance is not None: out.append(instance)
    if len(out)==1: out=out[0]

    return out

def do_test_augment_flip_transpose(image, proposal=None, type=0):
    #image_show('image_before',image)
    height,width = image.shape[:2]
    image = do_flip_transpose(height,width, image=image, type=type)

    if proposal is not None:
        proposal = do_flip_transpose(height,width, proposal=proposal, type=type)

    return do_test_augment_identity(image, proposal)




def undo_test_augment_flip_transpose(image, type=0):
    height,width = image.shape[:2]
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width

    dummy_image = np.zeros((H,W,3),np.uint8)
#     rcnn_proposal, detection, mask, instance = undo_test_augment_identity(net, dummy_image)
    detection, mask, instance = do_flip_transpose(H,W, proposal=detection, mask=mask, instance=instance, type=t)
    rcnn_proposal             = do_flip_transpose(H,W, proposal=rcnn_proposal, type=t)

    return rcnn_proposal, detection, mask, instance


def do_test_augment_identity(image, proposal=None):
    height,width = image.shape[:2]
    h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    dx = w-width
    dy = h-height

    image = cv2.copyMakeBorder(image, left=0, top=0, right=dx, bottom=dy,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    if proposal is not None:
        h,w = image.shape[:2]
        proposal = proposal.copy()
        x1,y1 = proposal[:,3],proposal[:,4]
        x1[np.where(x1>width -1-dx)[0]]=w-1 #dx
        y1[np.where(y1>height-1-dy)[0]]=h-1 #dy
        proposal[:,3] = x1
        proposal[:,4] = y1
        return image, proposal

    else:
        return image