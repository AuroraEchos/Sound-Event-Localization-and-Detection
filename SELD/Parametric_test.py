import torch
import numpy as np
import pickle
import argparse
import os
import torch.utils.data as utils

from torchinfo import summary
from model import SELD_Model

def readFile(path):
    
    with open(path, 'r') as f:
        r = f.read()
        r = r.replace('=', '+').replace('\n', '+').split('+')
        new_r = []
        for i in r:
            if i=='True':
                new_r.append('1')
            elif i == 'False':
                new_r.append(0)
            elif i !='' and '#' not in i:
                new_r.append(i)
    # print(new_r)
    return new_r

def main(args):

    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #LOAD DATASET
    print ('\nLoading dataset')

    with open(args.training_predictors_path, 'rb') as f:
        training_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    with open(args.validation_predictors_path, 'rb') as f:
        validation_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)

    phase_string='_Phase' if args.phase else ''
    dataset_string='L3DAS21_'+str(args.n_mics)+'Mics_Magnidute'+phase_string+'_'+str(args.input_channels)+'Ch'
    #####################################NORMALIZATION####################################
    if args.dataset_normalization not in {'False','false','None','none'}:
        print('\nDataset_Normalization')
        if args.dataset_normalization in{'DQ_Normalization','UnitNormNormalization','UnitNorm'}:
        
            training_predictors = torch.tensor(training_predictors)
            training_target = torch.tensor(training_target)
            validation_predictors = torch.tensor(validation_predictors)
            validation_target = torch.tensor(validation_target)
            test_predictors = torch.tensor(test_predictors)
            test_target = torch.tensor(test_target)
            if args.n_mics==2:
                if args.domain in ['DQ','dq','dQ','Dual_Quaternion','dual_quaternion']:
                    dataset_string+=' Dataset Normalization for 2Mic 8Ch Magnitude Dual Quaternion UnitNorm'
                    print('Dataset Normalization for 2Mic 8Ch Magnitude Dual Quaternion UnitNorm')
                    ## TRAINING PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(training_predictors[:,:8,:,:], chunks=8, dim=1)
                    denominator_0 = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
                    denominator_1 = torch.sqrt(denominator_0)
                    deno_cross = q_0 * p_0 + q_1 * p_1 + q_2 * p_2 + q_3 * p_3

                    p_0 = p_0 - deno_cross / denominator_0 * q_0
                    p_1 = p_1 - deno_cross / denominator_0 * q_1
                    p_2 = p_2 - deno_cross / denominator_0 * q_2
                    p_3 = p_3 - deno_cross / denominator_0 * q_3

                    q_0 = q_0 / denominator_1
                    q_1 = q_1 / denominator_1
                    q_2 = q_2 / denominator_1
                    q_3 = q_3 / denominator_1

                    training_predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1)

                    ## VALIDATION PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(validation_predictors[:,:8,:,:], chunks=8, dim=1)
                    denominator_0 = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
                    denominator_1 = torch.sqrt(denominator_0)
                    deno_cross = q_0 * p_0 + q_1 * p_1 + q_2 * p_2 + q_3 * p_3

                    p_0 = p_0 - deno_cross / denominator_0 * q_0
                    p_1 = p_1 - deno_cross / denominator_0 * q_1
                    p_2 = p_2 - deno_cross / denominator_0 * q_2
                    p_3 = p_3 - deno_cross / denominator_0 * q_3

                    q_0 = q_0 / denominator_1
                    q_1 = q_1 / denominator_1
                    q_2 = q_2 / denominator_1
                    q_3 = q_3 / denominator_1

                    validation_predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1)

                    ## TEST PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(test_predictors[:,:8,:,:], chunks=8, dim=1)
                    denominator_0 = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
                    denominator_1 = torch.sqrt(denominator_0)
                    deno_cross = q_0 * p_0 + q_1 * p_1 + q_2 * p_2 + q_3 * p_3

                    p_0 = p_0 - deno_cross / denominator_0 * q_0
                    p_1 = p_1 - deno_cross / denominator_0 * q_1
                    p_2 = p_2 - deno_cross / denominator_0 * q_2
                    p_3 = p_3 - deno_cross / denominator_0 * q_3

                    q_0 = q_0 / denominator_1
                    q_1 = q_1 / denominator_1
                    q_2 = q_2 / denominator_1
                    q_3 = q_3 / denominator_1

                    test_predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1) 
                    if args.phase:
                        raise ValueError('DATASET NORMALIZATION FOR PHASE DUAL QUATERNION NOT YET IMPLEMENTED')
                        print('Dataset Normalization for 2Mic 16Ch Magnitude-Phase Dual Quaternion ')
                    training_predictors = np.array(training_predictors)
                    training_target = np.array(training_target)
                    validation_predictors = np.array(validation_predictors)
                    validation_target = np.array(validation_target)
                    test_predictors = np.array(test_predictors)
                    test_target = np.array(test_target)

                    print ('\nShapes:')
                    print ('Training predictors: ', training_predictors.shape)
                    print ('Validation predictors: ', validation_predictors.shape)
                    print ('Test predictors: ', test_predictors.shape)
                    print ('Training target: ', training_target.shape)
                    print ('Validation target: ', validation_target.shape)
                    print ('Test target: ', test_target.shape)
        else:
            training_predictors = np.array(training_predictors)
            training_target = np.array(training_target)
            validation_predictors = np.array(validation_predictors)
            validation_target = np.array(validation_target)
            test_predictors = np.array(test_predictors)
            test_target = np.array(test_target)

            print ('\nShapes:')
            print ('Training predictors: ', training_predictors.shape)
            print ('Validation predictors: ', validation_predictors.shape)
            print ('Test predictors: ', test_predictors.shape)
            print ('Training target: ', training_target.shape)
            print ('Validation target: ', validation_target.shape)
            print ('Test target: ', test_target.shape)
            if args.n_mics==1:
                dataset_string+=' Dataset Normalization for 1Mic 4Ch Magnitude'
                print('Dataset Normalization for 1Mic 4Ch Magnitude')
                # Normalize training predictors with mean 0 and std 1
                train_mag_min = np.mean(training_predictors[:,:4,:,:])
                train_mag_std = np.std(training_predictors[:,:4,:,:])  
                training_predictors[:,:4,:,:] -= train_mag_min
                training_predictors[:,:4,:,:] /= train_mag_std
                # Normalize validation predictors with mean 0 and std 1
                val_mag_min = np.mean(validation_predictors[:,:4,:,:])
                val_mag_std = np.std(validation_predictors[:,:4,:,:])    
                validation_predictors[:,:4,:,:] -= val_mag_min
                validation_predictors[:,:4,:,:] /= val_mag_std
                # Normalize test predictors with mean 0 and std 1
                test_mag_min = np.mean(test_predictors[:,:4,:,:])
                test_mag_std = np.std(test_predictors[:,:4,:,:])    
                test_predictors[:,:4,:,:] -= test_mag_min
                test_predictors[:,:4,:,:] /= test_mag_std
                if args.phase:
                    dataset_string+=' Dataset Normalization for 1Mic 8Ch Magnitude-Phase'
                    print('Dataset Normalization for 1Mic 8Ch Magnitude-Phase')
                    train_phase_min = np.mean(training_predictors[:,4:,:,:])
                    train_phase_std = np.std(training_predictors[:,4:,:,:])
                    training_predictors[:,4:,:,:] -= train_phase_min
                    training_predictors[:,4:,:,:] /= train_phase_std
                    val_phase_min = np.mean(validation_predictors[:,4:,:,:])
                    val_phase_std = np.std(validation_predictors[:,4:,:,:])
                    validation_predictors[:,4:,:,:] -= val_phase_min
                    validation_predictors[:,4:,:,:] /= val_phase_std
                    test_phase_min = np.mean(test_predictors[:,4:,:,:])
                    test_phase_std = np.std(test_predictors[:,4:,:,:])
                    test_predictors[:,4:,:,:] -= test_phase_min
                    test_predictors[:,4:,:,:] /= test_phase_std
            if args.n_mics==2:
                
                dataset_string+=' Dataset Normalization for 2Mic 8Ch Magnitude'
                print('Dataset Normalization for 2Mic 8Ch Magnitude')
                # Normalize training predictors with mean 0 and std 1
                train_mag_min = np.mean(training_predictors[:,:8,:,:])
                train_mag_std = np.std(training_predictors[:,:8,:,:])  
                training_predictors[:,:8,:,:] -= train_mag_min
                training_predictors[:,:8,:,:] /= train_mag_std
                # Normalize validation predictors with mean 0 and std 1
                val_mag_min = np.mean(validation_predictors[:,:8,:,:])
                val_mag_std = np.std(validation_predictors[:,:8,:,:])    
                validation_predictors[:,:8,:,:] -= val_mag_min
                validation_predictors[:,:8,:,:] /= val_mag_std
                # Normalize test predictors with mean 0 and std 1
                test_mag_min = np.mean(test_predictors[:,:8,:,:])
                test_mag_std = np.std(test_predictors[:,:8,:,:])    
                test_predictors[:,:8,:,:] -= test_mag_min
                test_predictors[:,:8,:,:] /= test_mag_std
                if args.phase:
                
                    dataset_string+=' Dataset Normalization for 2Mic 16Ch Magnitude-Phase'
                    print('Dataset Normalization for 2Mic 16Ch Magnitude-Phase')
                    train_phase_min = np.mean(training_predictors[:,8:,:,:])
                    train_phase_std = np.std(training_predictors[:,8:,:,:])
                    training_predictors[:,8:,:,:] -= train_phase_min
                    training_predictors[:,8:,:,:] /= train_phase_std
                    val_phase_min = np.mean(validation_predictors[:,8:,:,:])
                    val_phase_std = np.std(validation_predictors[:,8:,:,:])
                    validation_predictors[:,8:,:,:] -= val_phase_min
                    validation_predictors[:,8:,:,:] /= val_phase_std
                    test_phase_min = np.mean(test_predictors[:,8:,:,:])
                    test_phase_std = np.std(test_predictors[:,8:,:,:])
                    test_predictors[:,8:,:,:] -= test_phase_min
                    test_predictors[:,8:,:,:] /= test_phase_std
    else:
        training_predictors = np.array(training_predictors)
        training_target = np.array(training_target)
        validation_predictors = np.array(validation_predictors)
        validation_target = np.array(validation_target)
        test_predictors = np.array(test_predictors)
        test_target = np.array(test_target)

        print ('\nShapes:')
        print ('Training predictors: ', training_predictors.shape)
        print ('Validation predictors: ', validation_predictors.shape)
        print ('Test predictors: ', test_predictors.shape)
        print ('Training target: ', training_target.shape)
        print ('Validation target: ', validation_target.shape)
        print ('Test target: ', test_target.shape)
    
    ###############################################################################
    features_dim = int(test_target.shape[-2] * test_target.shape[-1])

    #convert to tensor
    training_predictors = torch.tensor(training_predictors).float()
    validation_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    training_target = torch.tensor(training_target).float()
    validation_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()
    #build dataset from tensors
    tr_dataset = utils.TensorDataset(training_predictors, training_target)
    val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)
    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, 1, shuffle=False, pin_memory=True)#(test_dataset, args.batch_size, shuffle=False, pin_memory=True

    #LOAD MODEL
    n_time_frames = test_predictors.shape[-1]

    ######################################################################################################################
    model=SELD_Model(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels, output_classes=args.output_classes,
                 domain=args.domain, domain_classifier=args.domain_classifier,
                 cnn_filters=args.cnn_filters, kernel_size_cnn_blocks=args.kernel_size_cnn_blocks, pool_size=args.pool_size, pool_time=args.pool_time,
                 D=args.D, dilation_mode=args.dilation_mode,G=args.G, U=args.U, kernel_size_dilated_conv=args.kernel_size_dilated_conv,
                 spatial_dropout_rate=args.spatial_dropout_rate,V=args.V, V_kernel_size=args.V_kernel_size,
                 fc_layers=args.fc_layers, fc_activations=args.fc_activations, fc_dropout=args.fc_dropout, dropout_perc=args.dropout_perc, 
                 class_overlaps=args.class_overlaps,
                 use_bias_conv=args.use_bias_conv,use_bias_linear=args.use_bias_linear,batch_norm=args.batch_norm,  parallel_ConvTC_block=args.parallel_ConvTC_block, parallel_magphase=args.parallel_magphase,
                 extra_name=args.model_extra_name, verbose=False)
    
                 
    architecture_dir='RESULTS/Task2/{}/'.format(args.architecture)
    if len(os.path.dirname(architecture_dir)) > 0 and not os.path.exists(os.path.dirname(architecture_dir)):
        os.makedirs(os.path.dirname(architecture_dir))
    model_dir=architecture_dir+model.model_name+'/'
    if len(os.path.dirname(model_dir)) > 0 and not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))
    args.load_model=model_dir+'checkpoint'
    unique_name=model_dir+model.model_name
    
    '''if not args.wandb_id=='none': 
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,resume='allow',id=args.wandb_id,name=model.model_name)############################################################################################ WANDB
    else:
        wandb.init(project=args.wandb_project,entity=args.wandb_entity,resume='allow',name=model.model_name)
    config = wandb.config
    wandb.watch(model)
    wandb.config.update(args, allow_val_change=True)
    wandb.config.ReceptiveField=model.receptive_field
    wandb.config.n_ResBlocks=model.total_n_resblocks'''
    
    print(dataset_string)
    print(model.model_name)
    
    summary(model, input_size=(args.batch_size,args.input_channels,args.freq_dim,n_time_frames)) ##################################################
    if not args.architecture == 'seldnet_vanilla' and not args.architecture == 'seldnet_augmented': 
        print('\nReceptive Field: ',model.receptive_field,'\nNumber of ResBlocks: ', model.total_n_resblocks)
    #######################################################################################################################
    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #saving/loading parameters
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default='RESULTS/Task2',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,#'RESULTS/Task2/checkpoint',
                        help='Reload a previously trained model (whole task model)')
    #dataset parameters
    parser.add_argument('--training_predictors_path', type=str,default='/var/datasets/L3DAS21/processed/task2_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str,default='/var/datasets/L3DAS21/processed/task2_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='/var/datasets/L3DAS21/processed/task2_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='/var/datasets/L3DAS21/processed/task2_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='/var/datasets/L3DAS21/processed/task2_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='/var/datasets/L3DAS21/processed/task2_target_test.pkl')
    #training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='True')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=32000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=250,
                        help="Patience for early stopping on validation set")

    #model parameters
    #the following parameters produce a prediction for each 100-msecs frame
    parser.add_argument('--architecture', type=str, default='DualQSELD-TCN',
                        help="model's architecture, can be seldnet_vanilla or seldnet_augmented")
    parser.add_argument('--input_channels', type=int, default=4,
                        help="4/8 for 1/2 mics, multiply x2 if using also phase information")
    parser.add_argument('--n_mics', type=int, default=1)
    parser.add_argument('--phase', type=str, default='False')
    parser.add_argument('--class_overlaps', type=int, default=3,
                        help= 'max number of simultaneous sounds of the same class')
    parser.add_argument('--time_dim', type=int, default=4800)
    parser.add_argument('--freq_dim', type=int, default=256)
    parser.add_argument('--output_classes', type=int, default=14)
    parser.add_argument('--pool_size', type=str, default='[[8,2],[8,2],[2,2],[1,1]]')
    parser.add_argument('--cnn_filters', type=str, default='[64,64,64]')
    parser.add_argument('--pool_time', type=str, default='True')
    parser.add_argument('--dropout_perc', type=float, default=0.3)
    parser.add_argument('--D', type=str, default='[10]')
    parser.add_argument('--G', type=int, default=128)
    parser.add_argument('--U', type=int, default=128)
    parser.add_argument('--V', type=str, default='[128,128]')
    parser.add_argument('--spatial_dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=str, default='BN')
    parser.add_argument('--dilation_mode', type=str, default='fibonacci')
    parser.add_argument('--model_extra_name', type=str, default='')
    parser.add_argument('--test_mode', type=str, default='test_best')
    parser.add_argument('--use_lr_scheduler', type=str, default='True')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=150)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=0.000005) 
    parser.add_argument('--dataset_normalization', type=str, default='True') 
    parser.add_argument('--kernel_size_cnn_blocks', type=int, default=3) 
    parser.add_argument('--kernel_size_dilated_conv', type=int, default=3) 
    parser.add_argument('--use_tcn', type=str, default='True') 
    parser.add_argument('--use_bias_conv', type=str, default='True') 
    parser.add_argument('--use_bias_linear', type=str, default='True') 
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--sed_loss_weight', type=float, default=1.)
    parser.add_argument('--doa_loss_weight', type=float, default=5.)
    parser.add_argument('--domain_classifier', type=str, default='same') 
    parser.add_argument('--domain', type=str, default='DQ') 
    parser.add_argument('--fc_activations', type=str, default='Linear') 
    parser.add_argument('--fc_dropout', type=str, default='Last') 
    parser.add_argument('--fc_layers', type=str, default='[128]') 
    parser.add_argument('--V_kernel_size', type=int, default=3) 
    parser.add_argument('--use_time_distributed', type=str, default='False') 
    parser.add_argument('--parallel_ConvTC_block', type=str, default='False') 

    '''parser.add_argument('--wandb_id', type=str, default='none')
    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--wandb_entity', type=str, default='')'''
    ############## TEST  ###################
    parser.add_argument('--max_loc_value', type=float, default=2.,
                         help='max value of target loc labels (to rescale model\'s output since the models has tanh in the output loc layer)')
    parser.add_argument('--num_frames', type=int, default=600,
                        help='total number of time frames in the predicted seld matrices. (600 for 1-minute sounds with 100msecs frames)')
    parser.add_argument('--spatial_threshold', type=float, default=2.,
                        help='max cartesian distance withn consider a true positive')
    ########################################

    ######################### CHECKPOINT ####################################################
    parser.add_argument('--checkpoint_step', type=int, default=100,
                        help="Save and test models every checkpoint_step epochs")
    parser.add_argument('--test_step', type=int, default=10,
                        help="Save and test models every checkpoint_step epochs")
    parser.add_argument('--min_n_epochs', type=int, default=1000,
                        help="Save and test models every checkpoint_step epochs")
    parser.add_argument('--Dcase21_metrics_DOA_threshold', type=int, default=20) 
    parser.add_argument('--parallel_magphase', type=str, default='False') 

    parser.add_argument('--TextArgs', type=str, default='config/Test.txt', help='Path to text with training settings')#'config/PHC-SELD-TCN-S1_BN.txt'
    parse_list = readFile(parser.parse_args().TextArgs)
    args = parser.parse_args(parse_list)
    
    #eval string bools and lists
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)
    args.pool_size= eval(args.pool_size)
    args.cnn_filters = eval(args.cnn_filters)
    args.verbose = eval(args.verbose)
    args.D=eval(args.D)
    args.V=eval(args.V)
    args.use_lr_scheduler=eval(args.use_lr_scheduler)
    #args.dataset_normalization=eval(args.dataset_normalization)
    args.phase=eval(args.phase)
    args.use_tcn=eval(args.use_tcn)
    args.use_bias_conv=eval(args.use_bias_conv)
    args.use_bias_linear=eval(args.use_bias_linear)
    args.fc_layers = eval(args.fc_layers)
    args.parallel_magphase = eval(args.parallel_magphase)

    main(args)