# Training loop
from data_load_utils import *
import time 
batch_size = 10 # the CamVid training set only contains 367 samples 
max_epochs = 10
current_epoch=0
# Keep track of losses and accuracies
validation_loss, validation_accuracy = [], []
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []
_train_loss, _train_accuracy = [], []

# Load validation data
vlocx, vlocy = parsepaths(valid_paths)
x_validation, y_validation = loadimages(vlocx, vlocy)
           
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Start')
    starttime=time.time()
    try: 
        while current_epoch<max_epochs:
            # Intialize empty looplists
            # Next batch
            x_batch, y_batch = nextbatch(x_im, y_im, current_epoch,batch_size)
            # What to fetch and what to feed
            fetches_train = [train_op, cross_entropy, accuracy]
            feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
            # Feed and fetch
            _, _loss, _acc = sess.run(fetches_train, feed_dict_train)
        
            _train_loss.append(_loss)
            _train_accuracy.append(_acc)
            
            current_epoch+=1
            
            # Validation
            if(current_epoch%10==0):
                train_loss.append(np.mean(_train_loss))
                train_accuracy.append(np.mean(_train_accuracy))
                fetches_validation = [train_op, cross_entropy, accuracy]
                feed_dict_validation = {x_pl: x_validation, y_pl: y_validation}
                _loss, _acc = sess.run(fetches_validation, feed_dict_validation)
                validation_loss.append(_loss)
                validation_accuracy.append(_acc)
                _train_loss, _train_accuracy = [], []
                print("Epoch: %s \n  Train Loss %s, Train acc %s, \nValidation loss %s,  Validation acc %s"%(current_epoch, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1]))           
    except KeyboardInterrupt:
        pass 

print("Done in %s seconds"%(time.time()-starttime))