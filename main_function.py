import netCDF4 as nc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import hydroeval as he
import shap
import gc
import h5py

#The following code is the XAI framework.
#vegctr is the sub-region used for training the model.
#"getncdata" is the function used to read the ".nc" file data; "fpin" is the file path; "prna" is the file name; "varna" is variable name; and "regionid" is the region identification code.
#"lstm_train" is the function used to train the model; "xy_train_test" is the training and validation datasets; "seqlen" is the number of time steps; "nfeatures" is the number of variables in the training dataset; "modelna" is the name for saving model; "fpmodout" is the file path for saving model.
#"loadmodel" is the function used to load the trained model from the "lstm_train"; "fpin" is the file path; "smna","prna","tasna", and "tranna" are the file names for soil moisture, precipitation, temperature, and transpiration, respectively; "mod_path" is the file path for saving model;
#"getnc" is similar to "getncdata"
###vegctr_id,fpin,smna,prna,tasna,tranna,mod_path,modelna,seqlen1


fpbase = '/lustre/home/'
vegctr = np.transpose(np.load(fpbase + 'lucc_1.5degree_vegetated_area.npy'))[::-1,:]
vegctr = vegctr.astype(float)
vegctr_id = np.unique(vegctr.flatten())
vegctr_id = vegctr_id[~np.isnan(vegctr_id)]

def getncdata(fpin, prna, varna, vegctr, regionid):
    loca_subregion = np.where(vegctr == regionid)
    x = [np.min(loca_subregion[0]), np.max(loca_subregion[0]) + 1]
    y = [np.min(loca_subregion[1]), np.max(loca_subregion[1]) + 1]
    #prna = 'precipitation_gldas_clsm_19480101_20221231_1.5degree_pentad.nc'
    prinfo = nc.Dataset(fpin + prna)
    prda = prinfo[varna][:][:, ::-1, :]
    prda = prda[:,x[0]:x[1],y[0]:y[1]]
    prinfo.close()
    # plt.imshow(prda[1, :, :])
    # plt.colorbar(label='Color Scale')
    # plt.show()
    return prda.data

def lstm_train(xy_train_test,seqlen,nfeatures,regioni,modelna,fpmodout):
   #xy_train_test = [x_train_all, y_train_all, x_val_all, y_val_all, rs, trainseq, testseq]
    epochi = 50
    batch_size = 528

    model = Sequential()
    model.add(LSTM(units=30, input_shape=(seqlen, nfeatures)))
    model.add(Dense(units=1))
    model.add(Dropout(rate=0.1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics='mse')
    es_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1)

    with tf.device('/gpu:0'):
        model.fit(xy_train_test[0], xy_train_test[1], epochs=epochi, batch_size=batch_size, verbose=0,
                  callbacks=[es_callback],
                  validation_data=(xy_train_test[2], xy_train_test[3]))

    y_test_all = xy_train_test[3].reshape(-1)
    trainPredict = model.predict(xy_train_test[2])
    NSE_test = he.evaluator(he.nse, trainPredict.reshape(-1), y_test_all)[0]
    KGE_test = he.evaluator(he.kge, trainPredict.reshape(-1), y_test_all)[0, 0]

   #modelna = 'A'; regioni = 'B'; seqlen='36'
    print('NSE value: ' + str(NSE_test))
    print('KGE value: ' + str(KGE_test))
    h5_name = fpmodout + f"model_{modelna}_region_{regioni}_time_{seqlen}_10fold.h5"
    model.save(h5_name)

    return NSE_test, KGE_test
def loadmodel(vegctr_id,fpin,smna,prna,tasna,tranna,mod_path,modelna,seqlen):


    def getncdata(fpin, varna, regionid, vegctr,ncna):
        loca_subregion = np.where(vegctr == regionid)
        x = [np.min(loca_subregion[0]), np.max(loca_subregion[0]) + 1]
        y = [np.min(loca_subregion[1]), np.max(loca_subregion[1]) + 1]
        prinfo = nc.Dataset(fpin + ncna)
        prda = prinfo[varna][:][:, ::-1, :]
        prda = prda[:, x[0]:x[1], y[0]:y[1]]
        prinfo.close()
        # plt.imshow(prda[1, :, :])
        # plt.show()
        return prda.data

    models = {}
    for mod_i in vegctr_id: #mod_i = 0.0

        smda   = getncdata(fpin, varna='sm'  , regionid=mod_i, vegctr=vegctr, ncna = smna  )
        prda   = getncdata(fpin, varna='pr'  , regionid=mod_i, vegctr=vegctr, ncna = prna  )
        tasda  = getncdata(fpin, varna='tas' , regionid=mod_i, vegctr=vegctr, ncna = tasna )
        tranda = getncdata(fpin, varna='tran', regionid=mod_i, vegctr=vegctr, ncna = tranna)
        x_train= []

       #model save path; ij_sta = 0.0
        h5_file_path = mod_path + f'model_{modelna}_region_{mod_i}_time_{seqlen}_10fold.h5'

       #with tf.device('CPU'):
        model = tf.keras.models.load_model(h5_file_path)

        explainer = shap.GradientExplainer(model, x_train)

       #save model into 'models', with mod_i as id
        models[mod_i] = explainer
        del explainer
        del model
        del x_train
        gc.collect()
       #print(mod_i)

    return models

def getnc(fpin, varna, ncna):
    prinfo = nc.Dataset(fpin + ncna)
    prda = prinfo[varna][:][:, ::-1, :]
    prinfo.close()
   #plt.imshow(prda[1, :, :])
   #plt.show()
    return prda.data

###The following code is used for calculating EG values.

modnaall = [] #model names, such as "ACCESS-ESM1-5-all"
moddataall = [] #data names, such as "ACCESS-ESM1-5"
fp_mod_path = f'/lustre/' # model path
fp_data = '/lustre/' #data path
fpout = '/lustre/'  #path for saving EG values
seqlen = 36
for ii in np.arange(0, len(modnaall)):  # mod='ACCESS-ESM1-5'

    mod = modnaall[ii]
    moddai = moddataall[ii]

    outna = fpout + 'shap_36ts_' + mod + '.hdf5'
    f = h5py.File(outna, 'w')
    shape = (93, 240, 5329, 36, 3)
    dset = f.create_dataset("data", shape, dtype=np.float32, fillvalue=np.nan, compression="gzip", compression_opts=3)

    fd_location = np.load(fp_data + 'fd_event_location.npy')
    fp_location = np.transpose(fd_location, (0, 2, 1))[:, ::-1, :]

    smna1 = 'mrsol.nc'
    prna1 = 'pr.nc'
    tasna1 = 'tas.nc'
    tranna1 = 'tran.nc'

    models = loadmodel(vegctr_id=vegctr_id,
                       fpin=fp_data,
                       smna=smna1,
                       prna=prna1,
                       tasna=tasna1,
                       tranna=tranna1,
                       mod_path=fp_mod_path,
                       modelna=mod,
                       seqlen=seqlen)

    smda = getnc(fp_data, varna='sm', ncna=smna1)
    prda = getnc(fp_data, varna='pr', ncna=prna1)
    tasda = getnc(fp_data, varna='tas', ncna=tasna1)
    tranda = getnc(fp_data, varna='tran', ncna=tranna1)

    nfeatures = 3
    for i in range(0, vegctr.shape[0]):
        for j in range(0, vegctr.shape[1]):  # i=38;j=190
            if not np.isnan(vegctr[i, j]):

                ij_sta = vegctr[i, j]
                modeli = models[ij_sta]

                x_all = [] #that is x variable for lstm;
                daid = []#the location needed to calculate EG values;
                if len(daid) > 0:
                    fp_location_da = x_all[daid, :, :]
                    fp_shap_value = modeli.shap_values(fp_location_da)[0]
                    fp_shap_all = np.full(((tasda.shape[0]), seqlen, nfeatures), np.nan)
                    fp_shap_all[daid, :, :] = fp_shap_value

                    dset[i, j, :, :, :] = fp_shap_all

    f.close()
    print(mod)
    del models
    del smda
    del prda
    del tasda
    del tranda
    gc.collect()