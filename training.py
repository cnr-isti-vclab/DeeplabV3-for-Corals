import sys
import os
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from deeplab import DeepLab
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from coral_dataset import CoralsDataset
from labelsdictionary import dictScripps as dictionary
import json
import shutil
from torch.utils.tensorboard import SummaryWriter
import losses
from torch.autograd import Variable
import pandas as pd

# SEED
torch.manual_seed(997)
np.random.seed(997)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def saveMetrics(metrics, filename):
    """
    Save the computed metrics.
    """

    file = open(filename, 'w')

    file.write("CONFUSION MATRIX: \n\n")

    np.savetxt(file, metrics['ConfMatrix'], fmt='%d')
    file.write("\n")

    file.write("NORMALIZED CONFUSION MATRIX: \n\n")

    np.savetxt(file, metrics['NormConfMatrix'], fmt='%.3f')
    file.write("\n")

    file.write("ACCURACY      : %.3f\n\n" % metrics['Accuracy'])
    file.write("Jaccard Score : %.3f\n\n" % metrics['JaccardScore'])

    file.close()


# VALIDATION
def evaluateNetwork(dataset, dataloader, loss_to_use, CEloss, w_for_GDL, tversky_loss_alpha, tversky_loss_beta,
                    focal_tversky_gamma, epoch, epochs_switch, epochs_transition, nclasses, net,
                    flag_compute_mIoU=False, savefolder=""):
    """
    It evaluates the network on the validation set.  
    :param dataloader: Pytorch DataLoader to load the dataset for the evaluation.
    :param net: Network to evaluate.
    :param savefolder: if a folder is given the classification results are saved into this folder. 
    :return: all the computed metrics.
    """""

    ##### SETUP THE NETWORK #####

    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        device = torch.device("cuda")
        net.to(device)
        torch.cuda.synchronize()

    ##### EVALUATION #####

    net.eval()  # set the network in evaluation mode

    batch_size = dataloader.batch_size

    CM = np.zeros((nclasses, nclasses), dtype=int)
    class_indices = list(range(nclasses))

    ypred_list = []
    ytrue_list = []
    loss_values = []
    with torch.no_grad():
        for k, data in enumerate(dataloader):

            batch_images, labels_batch, names = data['image'], data['labels'], data['name']
            print(names)

            if USE_CUDA:
                batch_images = batch_images.to(device)
                labels_batch = labels_batch.to(device)

            # N x K x H x W --> N: batch size, K: number of classes, H: height, W: width
            outputs = net(batch_images)

            # predictions size --> N x H x W
            values, predictions_t = torch.max(outputs, 1)

            if loss_to_use == "NONE":
                loss_values.append(0.0)
            else:
                loss = computeLoss(loss_to_use, CEloss, w_for_GDL, tversky_loss_alpha, tversky_loss_beta,
                                   focal_tversky_gamma, epoch, epochs_switch, epochs_transition, labels_batch, outputs)

                loss_values.append(loss.item())

            pred_cpu = predictions_t.cpu()
            labels_cpu = labels_batch.cpu()

            if flag_compute_mIoU:
                ypred_list.extend(pred_cpu.numpy().ravel())
                ytrue_list.extend(labels_cpu.numpy().ravel())

            # CONFUSION MATRIX, PREDICTIONS ARE PER-COLUMN, GROUND TRUTH CLASSES ARE PER-ROW
            for i in range(batch_size):
                print(i)
                pred_index = pred_cpu[i].numpy().ravel()
                true_index = labels_cpu[i].numpy().ravel()
                confmat = confusion_matrix(true_index, pred_index, class_indices)
                CM += confmat

            # SAVE THE OUTPUT OF THE NETWORK
            for i in range(batch_size):

                if savefolder:
                    imgfilename = os.path.join(savefolder, names[i])
                    dataset.saveClassificationResult(batch_images[i].cpu(), outputs[i].cpu(), imgfilename)

    mean_loss = sum(loss_values) / len(loss_values)

    jaccard_s = 0.0

    if flag_compute_mIoU:
        ypred = np.array(ypred_list)
        del ypred_list
        ytrue = np.array(ytrue_list)
        del ytrue_list
        jaccard_s = jaccard_score(ytrue, ypred, average='weighted')

    # NORMALIZED CONFUSION MATRIX
    sum_row = CM.sum(axis=1)
    sum_row = sum_row.reshape((nclasses, 1))   # transform into column vector
    CMnorm = CM / sum_row    # divide each row using broadcasting


    # FINAL ACCURACY
    pixels_total = CM.sum()
    pixels_correct = np.sum(np.diag(CM))
    accuracy = float(pixels_correct) / float(pixels_total)


    metrics = {'ConfMatrix': CM, 'NormConfMatrix': CMnorm, 'Accuracy': accuracy, 'JaccardScore': jaccard_s}

    return metrics, mean_loss


def readClassifierInfo(filename, dataset):

    f = open(filename, "r")
    try:
        loaded_dict = json.load(f)
    except json.JSONDecodeError as e:
        print("File not found (!)")
        return

    dataset.num_classes = loaded_dict["Num. Classes"]
    dataset.weights = np.array(loaded_dict["Weights"])
    dataset.dataset_average = np.array(loaded_dict["Average"])
    dataset.dict_target = loaded_dict["Classes"]


def writeClassifierInfo(filename, classifier_name, dataset):

    dict_to_save = {}

    dict_to_save["Classifier Name"] = classifier_name
    dict_to_save["Weights"] = list(dataset.weights)
    dict_to_save["Average"] = list(dataset.dataset_average)
    dict_to_save["Num. Classes"] = dataset.num_classes
    dict_to_save["Classes"] = dataset.dict_target


    str = json.dumps(dict_to_save)

    f = open(filename, "w")
    f.write(str)
    f.close()


def computeLoss(loss_name, CE, w_for_GDL, tversky_alpha, tversky_beta, focal_tversky_gamma,
                epoch, epochs_switch, epochs_transition, labels, predictions):
    """
    Compute the loss given its name.
    """

    if loss_name == "CROSSENTROPY":
        loss = CE(predictions, labels)
    elif loss_name == "DICE":
        loss = losses.GDL(predictions, labels, w_for_GDL)
    elif loss_name == "BOUNDARY":
        loss = losses.surface_loss(labels, predictions)
    elif loss_name == "DICE+BOUNDARY":
        if epoch >= epochs_switch:
            alpha = 1.0 - (float(epoch - epochs_switch) / float(epochs_transition))
            if alpha < 0.0:
                alpha = 0.0
            loss = alpha * losses.GDL(predictions, labels, w_for_GDL) + (1.0 - alpha) * losses.surface_loss(labels, predictions)
        else:
            loss = losses.GDL(predictions, labels, w_for_GDL)
    elif loss_name == "FOCAL TVERSKY":
        loss = losses.focal_tversky(predictions, labels, tversky_alpha, tversky_beta, focal_tversky_gamma)
    elif loss_name == "FOCAL+BOUNDARY":
        if epoch >= epochs_switch:
            alpha = 1.0 - (float(epoch - epochs_switch) / float(epochs_transition))
            if alpha < 0.0:
                alpha = 0.0
            loss = alpha * losses.focal_tversky(predictions, labels, tversky_alpha, tversky_beta,
                                                focal_tversky_gamma) + (1.0 - alpha) * losses.surface_loss(labels, predictions)
        else:
            loss = losses.focal_tversky(predictions, labels, tversky_alpha, tversky_beta, focal_tversky_gamma)

    return loss

def computeBoundaryLossRange(images_folder_train, labels_folder_train, images_folder_val, labels_folder_val,
                    dictionary, target_classes, num_classes, save_network_as, save_classifier_as, classifier_name,
                    epochs, batch_sz, batch_mult, learning_rate, L2_penalty, validation_frequency, loss_to_use,
                    epochs_switch, epochs_transition, tversky_alpha, tversky_gamma, optimiz, flagShuffle, experiment_name):

    ##### DATA #####

    # setup the training dataset
    datasetTrain = CoralsDataset(images_folder_train, labels_folder_train, dictionary, target_classes, num_classes)

    print("Dataset setup..", end='')
    datasetTrain.computeAverage()
    datasetTrain.computeWeights()
    print(datasetTrain.dict_target)
    print(datasetTrain.weights)
    freq = 1.0 / datasetTrain.weights
    print(freq)
    print("done.")

    writeClassifierInfo(save_classifier_as, classifier_name, datasetTrain)

    datasetTrain.enableAugumentation()

    datasetVal = CoralsDataset(images_folder_val, labels_folder_val, dictionary, target_classes, num_classes)
    datasetVal.dataset_average = datasetTrain.dataset_average
    datasetVal.weights = datasetTrain.weights

    #AUGUMENTATION IS NOT APPLIED ON THE VALIDATION SET
    datasetVal.disableAugumentation()

    # setup the data loader
    dataloaderTrain = DataLoader(datasetTrain, batch_size=batch_sz, shuffle=flagShuffle, num_workers=0, drop_last=True,
                                 pin_memory=True)

    validation_batch_size = 4
    dataloaderVal = DataLoader(datasetVal, batch_size=validation_batch_size, shuffle=False, num_workers=0, drop_last=True,
                                 pin_memory=True)

    training_images_number = len(datasetTrain.images_names)
    validation_images_number = len(datasetVal.images_names)

    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        device = torch.device("cuda")

    print("Evaluate Boundary Loss range:")
    loss_values = []
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, minibatch in enumerate(dataloaderVal):
            # get the inputs
            images_batch = minibatch['image']
            labels_batch = minibatch['labels']

            if USE_CUDA:
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)

            loss = losses.surface_loss_fake(labels_batch, num_classes)

            loss_values.append(loss.item())
            print(loss.item())

    print("Min:", min(loss_values))
    print("Max:", max(loss_values))

    loss_values = []
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, minibatch in enumerate(dataloaderTrain):
            # get the inputs
            images_batch = minibatch['image']
            labels_batch = minibatch['labels']

            if USE_CUDA:
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)

            loss = losses.surface_loss_fake(labels_batch, num_classes)

            loss_values.append(loss.item())
            print(loss.item())

    print("Min:", min(loss_values))
    print("Max:", max(loss_values))


def trainingNetwork(images_folder_train, labels_folder_train, images_folder_val, labels_folder_val,
                    dictionary, target_classes, num_classes, save_network_as, save_classifier_as, classifier_name,
                    epochs, batch_sz, batch_mult, learning_rate, L2_penalty, validation_frequency, loss_to_use,
                    epochs_switch, epochs_transition, tversky_alpha, tversky_gamma, optimiz, flagShuffle, experiment_name):

    ##### DATA #####

    # setup the training dataset
    datasetTrain = CoralsDataset(images_folder_train, labels_folder_train, dictionary, target_classes, num_classes)

    print("Dataset setup..", end='')
    datasetTrain.computeAverage()
    datasetTrain.computeWeights()
    print(datasetTrain.dict_target)
    print(datasetTrain.weights)
    freq = 1.0 / datasetTrain.weights
    print(freq)
    print("done.")

    writeClassifierInfo(save_classifier_as, classifier_name, datasetTrain)

    datasetTrain.enableAugumentation()

    datasetVal = CoralsDataset(images_folder_val, labels_folder_val, dictionary, target_classes, num_classes)
    datasetVal.dataset_average = datasetTrain.dataset_average
    datasetVal.weights = datasetTrain.weights

    #AUGUMENTATION IS NOT APPLIED ON THE VALIDATION SET
    datasetVal.disableAugumentation()

    # setup the data loader
    dataloaderTrain = DataLoader(datasetTrain, batch_size=batch_sz, shuffle=flagShuffle, num_workers=0, drop_last=True,
                                 pin_memory=True)

    validation_batch_size = 4
    dataloaderVal = DataLoader(datasetVal, batch_size=validation_batch_size, shuffle=False, num_workers=0, drop_last=True,
                                 pin_memory=True)

    training_images_number = len(datasetTrain.images_names)
    validation_images_number = len(datasetVal.images_names)

    print("NETWORK USED: DEEPLAB V3+")

    if os.path.exists(save_network_as):
        net = DeepLab(backbone='resnet', output_stride=16, num_classes=datasetTrain.num_classes)
        net.load_state_dict(torch.load(save_network_as))
        print("Checkpoint loaded.")
    else:
        ###### SETUP THE NETWORK #####
        net = DeepLab(backbone='resnet', output_stride=16, num_classes=datasetTrain.num_classes)
        state = torch.load("deeplab-resnet.pth.tar")
        # RE-INIZIALIZE THE CLASSIFICATION LAYER WITH THE RIGHT NUMBER OF CLASSES, DON'T LOAD WEIGHTS OF THE CLASSIFICATION LAYER
        new_dictionary = state['state_dict']
        del new_dictionary['decoder.last_conv.8.weight']
        del new_dictionary['decoder.last_conv.8.bias']
        net.load_state_dict(state['state_dict'], strict=False)

    # OPTIMIZER
    if optimiz == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=L2_penalty, momentum=0.9)
    elif optimiz == "ADAM":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=L2_penalty)
    elif optimiz == "QHADAM":
        pass

    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        device = torch.device("cuda")
        net.to(device)

    ##### TRAINING LOOP #####

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(comment=experiment_name)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    best_accuracy = 0.0
    best_jaccard_score = 0.0


    # Crossentropy loss
    weights = datasetTrain.weights
    class_weights = torch.FloatTensor(weights).cuda()
    CEloss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    # weights for GENERALIZED DICE LOSS (GDL)
    freq = 1.0 / datasetTrain.weights[1:]
    w = 1.0 / (freq * freq)
    w = w / w.sum() + 0.00001
    w_for_GDL = torch.from_numpy(w)
    w_for_GDL = w_for_GDL.to(device)

    # Focal Tversky loss
    focal_tversky_gamma = torch.tensor(tversky_gamma)
    focal_tversky_gamma = focal_tversky_gamma.to(device)

    tversky_loss_alpha = torch.tensor(tversky_alpha)
    tversky_loss_beta = torch.tensor(1.0 - tversky_alpha)
    tversky_loss_alpha = tversky_loss_alpha.to(device)
    tversky_loss_beta = tversky_loss_beta.to(device)

    print("Training Network")
    for epoch in range(epochs):  # loop over the dataset multiple times

        net.train()
        optimizer.zero_grad()

        writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], epoch)

        running_loss = 0.0
        for i, minibatch in enumerate(dataloaderTrain):
            # get the inputs
            images_batch = minibatch['image']
            labels_batch = minibatch['labels']

            if USE_CUDA:
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)

            # forward+loss+backward
            outputs = net(images_batch)

            loss = computeLoss(loss_to_use, CEloss, w_for_GDL, tversky_loss_alpha, tversky_loss_beta, focal_tversky_gamma,
                               epoch, epochs_switch, epochs_transition, labels_batch, outputs)

            loss.backward()

            # TO AVOID MEMORY TROUBLE UPDATE WEIGHTS EVERY BATCH SIZE X BATCH MULT
            if (i+1)% batch_mult == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(epoch, i, loss.item())
            running_loss += loss.item()

        print("Epoch: %d , Running loss = %f" % (epoch, running_loss))

        ### VALIDATION ###
        if epoch > 0 and (epoch+1) % validation_frequency == 0:

            print("RUNNING VALIDATION.. ", end='')

            metrics_val, mean_loss_val = evaluateNetwork(datasetVal, dataloaderVal, loss_to_use, CEloss, w_for_GDL,
                                                         tversky_loss_alpha, tversky_loss_beta, focal_tversky_gamma,
                                                         epoch, epochs_switch, epochs_transition,
                                                         datasetVal.num_classes, net, flag_compute_mIoU=False)
            accuracy = metrics_val['Accuracy']
            jaccard_score = metrics_val['JaccardScore']

            scheduler.step(mean_loss_val)

            metrics_train, mean_loss_train = evaluateNetwork(datasetTrain, dataloaderTrain, loss_to_use, CEloss, w_for_GDL,
                                                             tversky_loss_alpha, tversky_loss_beta, focal_tversky_gamma,
                                                             epoch, epochs_switch, epochs_transition,
                                                             datasetTrain.num_classes, net, flag_compute_mIoU=False)
            accuracy_training = metrics_train['Accuracy']
            jaccard_training = metrics_train['JaccardScore']

            writer.add_scalar('Loss/train', mean_loss_train, epoch)
            writer.add_scalar('Loss/validation', mean_loss_val, epoch)
            writer.add_scalar('Accuracy/train', accuracy_training, epoch)
            writer.add_scalar('Accuracy/validation', accuracy, epoch)

            #if jaccard_score > best_jaccard_score:
            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_jaccard_score = jaccard_score
                torch.save(net.state_dict(), save_network_as)
                # performance of the best accuracy network on the validation dataset
                metrics_filename = save_network_as[:len(save_network_as) - 4] + "-val-metrics.txt"
                saveMetrics(metrics_val, metrics_filename)
                metrics_filename = save_network_as[:len(save_network_as) - 4] + "-train-metrics.txt"
                saveMetrics(metrics_train, metrics_filename)

            print("-> CURRENT BEST ACCURACY ", best_accuracy)


    # main loop ended - reload it and evaluate mIoU
    torch.cuda.empty_cache()
    del net
    net = None

    print("Final evaluation..")
    net = DeepLab(backbone='resnet', output_stride=16, num_classes=datasetTrain.num_classes)
    net.load_state_dict(torch.load(save_network_as))

    metrics_val, mean_loss_val = evaluateNetwork(datasetVal, dataloaderVal, loss_to_use, CEloss, w_for_GDL,
                                                 tversky_loss_alpha, tversky_loss_beta, focal_tversky_gamma,
                                                 epoch, epochs_switch, epochs_transition,
                                                 datasetVal.num_classes, net, flag_compute_mIoU=True)

    writer.add_hparams({'LR': learning_rate, 'Decay': L2_penalty, 'Loss': loss_to_use, 'Transition': epochs_transition,
                        'Gamma': tversky_gamma, 'Alpha': tversky_alpha }, {'hparam/Accuracy': best_accuracy, 'hparam/mIoU': best_jaccard_score})

    writer.close()

    print("***** TRAINING FINISHED *****")
    print("BEST ACCURACY REACHED ON THE VALIDATION SET: %.3f " % best_accuracy)


def testNetwork(images_folder, labels_folder, dictionary, target_classes, num_classes, classifier_info_filename, network_filename, output_folder):
    """
    Load a network and test it on the test dataset.g
    :param network_filename: Full name of the network to load (PATH+name)
    """


    # TEST DATASET
    datasetTest = CoralsDataset(images_folder, labels_folder, dictionary,  target_classes, num_classes)
    datasetTest.disableAugumentation()

    readClassifierInfo(classifier_info_filename, datasetTest)

    batchSize = 4
    dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=False, num_workers=0, drop_last=True,
                            pin_memory=True)
    # DEEPLAB V3+
    net = DeepLab(backbone='resnet', output_stride=16, num_classes=datasetTest.num_classes)
    net.load_state_dict(torch.load(network_filename))
    print("Weights loaded.")

    metrics_test, loss = evaluateNetwork(datasetTest, dataloaderTest, "NONE", None, [0.0], 0.0, 0.0, 0.0, 0, 0, 0,
                                         datasetTest.num_classes, net, False, output_folder)
    metrics_filename = network_filename[:len(network_filename) - 4] + "-test-metrics.txt"
    saveMetrics(metrics_test, metrics_filename)
    print("***** TEST FINISHED *****")


def main():

    # TARGET CLASSES
    target_classes = {"Background": 0,
                      "Pocillopora": 1,
                      "Porite_massive": 2,
                      "Montipora_plate/flabellata": 3,
                      "Montipora_crust/patula": 4,
                      "Montipora_capitata": 5
                      }

    # target_classes = {"Background": 0,
    #                   "Pocillopora": 1,
    #                   "Pocillopora_damicornis": 2,
    #                   "Pocillopora_zelli": 3,
    #                   "Pocillopora_eydouxi": 4,
    #                   "Porite_massive": 5,
    #                   "Montipora_plate/flabellata": 6,
    #                   "Montipora_crust/patula": 7,
    #                   "Montipora_capitata": 8
    #                   }



    # DATASET FOLDERS
    images_dir_train = "D:\\ten-orthos-scripps\\mini_train_im"
    labels_dir_train = "D:\\ten-orthos-scripps\\mini_train_lab"

    images_dir_val = "D:\\ten-orthos-scripps\\mini_val_im"
    labels_dir_val = "D:\\ten-orthos-scripps\\mini_val_lab"

    images_dir_test = "D:\\ten-orthos-scripps\\mini_test_im"
    labels_dir_test = "D:\\ten-orthos-scripps\\mini_test_lab"

    # LOAD EXPERIMENTS

    # LR = learning rate (0.00005)
    # L2 = weight decay (0.0005)
    # NEPOCHS = number of epochs
    # VAL_FREQ = validation frequency
    # BATCH_SIZE, BATCH_MULTIPLIER -> effective batch size = BATCH_SIZE * BATCH_MULTIPLIER
    #                                        the number of epochs for the transition is 1 / GDL_BOUNDARY_EPOCH_TRANSITION
    # LOSS_TO_USE -> loss to use
    #                "CROSSENTROPY"  -> Weighted Cross Entropy Loss
    #                "DICE"          -> Generalized Dice Loss (GDL)
    #                "BOUNDARY"      -> Boundary Loss
    #                "DICE+BOUNDARY" -> GDL, then Boundary Loss
    #                "FOCAL_TVERSKY" -> focal Tversky loss
    # GDL_BOUNDARY_EPOCH_SWITCH -> number of epochs before to switch to the Boundary loss (0 in the original implementation)
    # GDL_BOUNDARY_EPOCH_TRANSITION = 0.1 -> transition between GDL and BOUNDARY loss
    # TVERSKY_ALPHA -> IMPORTANCE OF FN w.r.t TP (0.7 REDUCES FN)
    # TVERSKY_GAMMA -> used by the FOCAL variant (>1 weights misclassified class more), 1/GAMMA in the original implementation
    # OPTIMIZER -> "Adam" or "SGD"
    #

    #experiments = pd.read_csv("experiments.csv")
    experiments = pd.read_csv("mini.csv")

    NCLASSES = len(target_classes)  # number of classes

    ##### RUN THE EXPERIMENTS
    for index, row in experiments.iterrows():

        LR = row["LR"]
        L2 = row["L2"]
        NEPOCHS = row["NEPOCHS"]
        VAL_FREQ = row["VAL_FREQ"]
        BATCH_SIZE = row["BATCH_SIZE"]
        BATCH_MULTIPLIER = row["BATCH_MULTIPLIER"]
        LOSS_TO_USE = row["LOSS_TO_USE"]
        # quando comincia a cambiare, 20 = alla epoch 20 comincia a cambiare
        GDL_BOUNDARY_EPOCH_SWITCH = row["GDL_BOUNDARY_EPOCH_SWITCH"]
        # in quante epoche completa la transizione (separate)
        GDL_BOUNDARY_EPOCH_TRANSITION = row["GDL_BOUNDARY_EPOCH_TRANSITION"]
        TVERSKY_ALPHA = row["TVERSKY_ALPHA"]
        TVERSKY_GAMMA = row["TVERSKY_GAMMA"]
        OPTIMIZER = row["OPTIMIZER"]

        params = "LR=" + str(LR) + "_L2=" + str(L2) + "_BS=" + str(BATCH_SIZE) + "x" + str(BATCH_MULTIPLIER) \
                 + "_loss=" + LOSS_TO_USE
        if LOSS_TO_USE == "DICE+BOUNDARY":
            params = params + "_SW=" + str(GDL_BOUNDARY_EPOCH_SWITCH) + "_TR=" + str(GDL_BOUNDARY_EPOCH_TRANSITION)
        elif LOSS_TO_USE == "FOCAL_TVERSKY":
            params = params + "_ALPHA=" + str(TVERSKY_ALPHA) + "_GAMMA=" + str(TVERSKY_GAMMA)

        params = params + "_OPT=" + OPTIMIZER

        network_name = "DEEPLAB_" + params + ".net"

        experiment_name = "_EXP_" + params

        save_classifier_as = network_name + ".json"
        classifier_name = "Coral 6-classes"

        ##### TRAINING
        trainingNetwork(images_dir_train, labels_dir_train, images_dir_val, labels_dir_val,
                        dictionary, target_classes, num_classes=NCLASSES, save_network_as=network_name,
                        save_classifier_as=save_classifier_as, classifier_name=classifier_name,
                        epochs=NEPOCHS, batch_sz=BATCH_SIZE, batch_mult=BATCH_MULTIPLIER,
                        validation_frequency=VAL_FREQ, loss_to_use=LOSS_TO_USE,
                        epochs_switch=GDL_BOUNDARY_EPOCH_SWITCH, epochs_transition=GDL_BOUNDARY_EPOCH_TRANSITION,
                        learning_rate=LR, L2_penalty=L2, tversky_alpha=TVERSKY_ALPHA, tversky_gamma=TVERSKY_GAMMA,
                        optimiz=OPTIMIZER, flagShuffle=True, experiment_name=experiment_name)

        # service function to compute the rangee of the Boundary loss on the training and the validation set
        # computeBoundaryLossRange(images_dir_train, labels_dir_train, images_dir_val, labels_dir_val,
        #                 dictionary, target_classes, num_classes=NCLASSES, save_network_as=network_name,
        #                 save_classifier_as=save_classifier_as, classifier_name=classifier_name,
        #                 epochs=NEPOCHS, batch_sz=BATCH_SIZE, batch_mult=BATCH_MULTIPLIER,
        #                 validation_frequency=VAL_FREQ, loss_to_use=LOSS_TO_USE,
        #                 epochs_switch=GDL_BOUNDARY_EPOCH_SWITCH, epochs_transition=GDL_BOUNDARY_EPOCH_TRANSITION,
        #                 learning_rate=LR, L2_penalty=L2, tversky_alpha=TVERSKY_ALPHA, tversky_gamma=TVERSKY_GAMMA,
        #                 optimiz=OPTIMIZER, flagShuffle=True, experiment_name=experiment_name)

        ##### TEST

        output_folder = os.path.join("temp", params)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        testNetwork(images_dir_test, labels_dir_test, dictionary,  target_classes, NCLASSES, save_classifier_as,
                    network_name, output_folder)


if __name__ == '__main__':
    main()
