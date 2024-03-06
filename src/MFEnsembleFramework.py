import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
from indago import DE, PSO
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error
import scipy.stats.qmc as qmc
import time

class InverseDesign:
    
    def __init__(self, input_data, 
                 output_data, 
                 forward_model, 
                 pca_model, 
                 evals,
                 top_rank,
                 optimizer='DE',
                 f_cut=None,test_size=1000):
        
        
        self.input_data = input_data
        self.output_data = output_data
        self.forward_model = forward_model
        self.pca_model = pca_model
        self.evals = evals
        self.top_rank = top_rank
        self.optimizer = optimizer
        self.f_cut = f_cut
        self.test_size=test_size
        
        
    def _load_models(self):
        
        return joblib.load(self.forward_model), joblib.load(self.pca_model)
    
    def _load_data(self):
        
        return np.load(self.input_data)[:self.test_size], np.load(self.output_data)[:self.test_size]
    
    def run(self):
        
        input_data, output_data = self._load_data()
        
        forward_model, pca_model = self._load_models()
        X = output_data
        
        predictions = []
        times = []
        prediction_sets = []
        c = 0
        for instance in input_data:
            print ('Instance: ', c)
            start = time.time()
            c+=1
            target = instance
            
            solutions_X = []
            solutions_f = []
            
            def objective_function(x):
                
                em = forward_model.predict(np.array(x).reshape(1,-1))
                em_val = pca_model.inverse_transform(em)
                rmse = np.sqrt(mean_squared_error(target, em_val[0]))   
                
                solutions_X.append(x)
                solutions_f.append(rmse)
                
                return rmse
            
            if self.optimizer == 'DE':
                optimizer = DE()
                optimizer.variant = 'LSHADE'    
                
                optimizer.lb = np.array([0.2, 10, 15])
                optimizer.ub = np.array([1.3, 700, 28])
                   
                optimizer.evaluation_function = objective_function
                if self.f_cut != None:
                    optimizer.target_fitness = self.f_cut
                optimizer.params['pop_init'] = 10
                optimizer.max_evaluations = self.evals
                optimizer.objectives = 1  
                #optimizer.monitoring = 'basic'
                result = optimizer.optimize()
                		
                min_f = result.f 
                min_x = result.X      
                
                predictions.append(min_x)
                            
                ind = np.argsort(solutions_f)[:self.top_rank]
                prediction_sets.append(np.array(solutions_X)[ind])
                end = np.abs(time.time() - start)
                times.append(end)
                
            elif self.optimizer == 'PSO':
                optimizer = PSO()
                
                optimizer.lb = np.array([0.2, 10, 15])
                optimizer.ub = np.array([1.3, 700, 28])
                   
                optimizer.evaluation_function = objective_function
                if self.f_cut != None:
                    optimizer.target_fitness = self.f_cut
                optimizer.params['swarm_size'] = 10
                optimizer.max_evaluations = self.evals
                optimizer.objectives = 1  
                #optimizer.monitoring = 'basic'
                result = optimizer.optimize()
                		
                min_f = result.f 
                min_x = result.X      
                
                predictions.append(min_x)
                            
                ind = np.argsort(solutions_f)[:self.top_rank]
                prediction_sets.append(np.array(solutions_X)[ind])

            
            elif self.optimizer == 'LBFGS':
                dim = 3
                min_f = 1
                min_x = None
                bounds = np.array([[0.2, 1.3], [10, 700], [15, 28]])

                for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, dim)):
                    res = opt.minimize(objective_function, x0=x0, bounds=bounds, method='LBFGS')        
                    if res.fun < min_f:
                        min_f = res.fun
                        min_x = res.x     
                        solutions_X.append(res.x)
                        solutions_f.append(res.fun)

            
            
                predictions.append(min_x)
                            
                ind = np.argsort(solutions_f)[:self.top_rank]
                prediction_sets.append(np.array(solutions_X)[ind])

        return np.array(predictions), prediction_sets, times
    
    
    def _get_ensemble_prediction(self, ensemble, instance):
        
        inputs_ = instance 
        
        averaged_prediction = ensemble.predict(inputs_[0].reshape(1,-1))
        
        def get_bagging_regressor_individual_predictions(bagging_regressor, X):
            individual_predictions = []
            for estimator in bagging_regressor.estimators_:
                individual_predictions.append(estimator.predict(X))
                
            return individual_predictions


        X_sample = inputs_[0].reshape(1,-1)
        all_predictions = []

        for bagging_regressor in ensemble.estimators_:
            all_predictions.append(get_bagging_regressor_individual_predictions(bagging_regressor, X_sample))

        parameters = []
        for i in range(len(all_predictions)):
            temp = []
            for j in range(len(all_predictions[i])):
                temp.append(all_predictions[i][j][0])
            parameters.append(temp)

        parameters = np.array(parameters)
        initial_sets = np.array([parameters[:,i] for i in range(np.shape(parameters)[1])])
        
        initial_set = np.vstack((initial_sets, averaged_prediction))
        
        return initial_sets#s, averaged_prediction
        
    
    def run_ensemble(self, ensemble_model_path):
        
        input_data, output_data = self._load_data()
        
        forward_model, pca_model = self._load_models()
        
        
        ensemble = joblib.load(ensemble_model_path)
        X = output_data
        
        predictions = []
        prediction_sets = []
        c = 1
        for instance in input_data:
            target = instance
            #print ('Instance: ', c)
            c += 1
   
            init_sets = self._get_ensemble_prediction(ensemble, pca_model.transform(instance.reshape(1,-1)))
            init_sets = np.unique(init_sets, axis=0)
         
            
            """end of ensemble code code"""
            
            solutions_X = []
            solutions_f = []
            
            def objective_function(x):
                
                em = forward_model.predict(np.array(x).reshape(1,-1))
                em_val = pca_model.inverse_transform(em)
                rmse = np.sqrt(mean_squared_error(target, em_val[0]))   
                
                solutions_X.append(x)
                solutions_f.append(rmse)
                
                return rmse

                
            optimizer = PSO()
            
            optimizer.lb = np.array([0.2, 10, 15])
            optimizer.ub = np.array([1.3, 700, 28])
            
            # optimizer.lb = np.array([np.min(init_sets[:,0]), np.min(init_sets[:,1]), np.min(init_sets[:,2])])
            # optimizer.ub = np.array([np.max(init_sets[:,0]), np.max(init_sets[:,1]), np.max(init_sets[:,2])])
            
            optimizer.X0 = init_sets
            optimizer.evaluation_function = objective_function
            if self.f_cut != None:
                optimizer.target_fitness = self.f_cut
            optimizer.params['swarm_size'] = 10
            optimizer.max_evaluations = self.evals
            optimizer.objectives = 1  
            #optimizer.monitoring = 'basic'
            result = optimizer.optimize()
            		
            min_f = result.f 
            min_x = result.X      
            
            predictions.append(min_x)
                        
            ind = np.argsort(solutions_f)[:self.top_rank]
            prediction_sets.append(np.array(solutions_X)[ind])
            
            
        return np.array(predictions), prediction_sets
    
    def run_RF_ensemble(self, ensemble_model_path):
        
        input_data, output_data = self._load_data()
        
        forward_model, pca_model = self._load_models()
        
        ensemble = joblib.load(ensemble_model_path)
        
        ensemble = joblib.load(ensemble_model_path)
        X = output_data
        
        times = []
        predictions = []
        prediction_sets = []
        c = 0
        for instance in input_data:
            target = instance
            start = time.time()
            print ('Instance: ', c)
            c += 1
   
            #init_sets = self._get_ensemble_prediction(ensemble, pca_model.transform(instance.reshape(1,-1)))
            
            
            #y_predict = ensemble.predict(pca_model.transform(instance.reshape(1,-1)))

            tree_predictions = []
            for tree in ensemble.estimators_:
                tree_predictions.append(tree.predict(pca_model.transform(instance.reshape(1,-1))))

            init_sets = np.vstack([i for i in tree_predictions])
            
            
            init_sets = np.unique(init_sets, axis=0)
            
            # for seed in init_sets:
            #     print (i)
        
            
            """end of ensemble code code"""
            
            solutions_X = []
            solutions_f = []
            
            def objective_function(x):
                
                em = forward_model.predict(np.array(x).reshape(1,-1))
                em_val = pca_model.inverse_transform(em)
                rmse = np.sqrt(mean_squared_error(target, em_val[0]))   
                
                solutions_X.append(x)
                solutions_f.append(rmse)
                
                return rmse

                
            optimizer = DE()
            
            optimizer.lb = np.array([0.2, 10, 15])
            optimizer.ub = np.array([1.3, 700, 28])
            
            #optimizer.lb = np.array([np.min(init_sets[:,0]), np.min(init_sets[:,1]), np.min(init_sets[:,2])])
            #optimizer.ub = np.array([np.max(init_sets[:,0]), np.max(init_sets[:,1]), np.max(init_sets[:,2])])
            
            optimizer.X0 = init_sets
            optimizer.evaluation_function = objective_function
            if self.f_cut != None:
                optimizer.target_fitness = self.f_cut
            #optimizer.params['swarm_size'] = 10
            optimizer.params['pop_init'] = 10
            optimizer.max_evaluations = self.evals
            optimizer.objectives = 1  
            #optimizer.monitoring = 'basic'
            result = optimizer.optimize()
            		
            min_f = result.f 
            min_x = result.X      
            end = np.abs(time.time() - start)
            times.append(end)
            predictions.append(min_x)
                        
            ind = np.argsort(solutions_f)[:self.top_rank]
            prediction_sets.append(np.array(solutions_X)[ind])
            
            
        return np.array(predictions), prediction_sets, times
    
    def run_direct(self, ensemble_model_path):
        
        input_data, output_data = self._load_data()
        
        forward_model, pca_model = self._load_models()
        
        ensemble = joblib.load(ensemble_model_path)
        
        ensemble = joblib.load(ensemble_model_path)
        X = output_data
        
        predictions = []
        times = []
        prediction_sets = []
        c = 1
        for instance in input_data:
            target = instance
            start = time.time()
            
            #print ('Instance: ', c)
            c += 1
   
            #init_sets = self._get_ensemble_prediction(ensemble, pca_model.transform(instance.reshape(1,-1)))
            tree_predictions = []
            for tree in ensemble.estimators_:
                tree_predictions.append(tree.predict(pca_model.transform(instance.reshape(1,-1))))
           
            init_sets = np.vstack([i for i in tree_predictions])
            #init_sets = np.unique(init_sets, axis=0)
            #print (len(init_sets))

           # ind = np.argsort(solutions_f)[:self.top_rank]
            #prediction_sets.append(init_sets.reshape(1,-1))
            end = np.abs(time.time() - start)
            times.append(end)
            prediction_sets.append(np.mean(init_sets.reshape(1,-1), axis=0))

            
        return np.array(prediction_sets), times
    
    
    def inference_RF(self, ensemble_model_path, target):
                
        #input_data, output_data = self._load_data()
        forward_model, pca_model = self._load_models()
        
        ensemble = joblib.load(ensemble_model_path)
                
        predictions = []
        prediction_sets = []
        
        tree_predictions = []
        for tree in ensemble.estimators_:
            tree_predictions.append(tree.predict(pca_model.transform(target)))

        init_sets = np.vstack([i for i in tree_predictions])
        init_sets = np.unique(init_sets, axis=0)
        
        
        

        """end of ensemble code code"""
        
        solutions_X = []
        solutions_f = []
        
        def objective_function(x):
            
            em = forward_model.predict(np.array(x).reshape(1,-1))
            em_val = pca_model.inverse_transform(em)
            rmse = np.sqrt(mean_squared_error(target, em_val))   
            
            solutions_X.append(x)
            solutions_f.append(rmse)
            
            return rmse

            
        optimizer = DE()
        
        optimizer.lb = np.array([0.2, 10, 15])
        optimizer.ub = np.array([1.3, 700, 28])
        
        optimizer.X0 = init_sets
        optimizer.evaluation_function = objective_function
        if self.f_cut != None:
            optimizer.target_fitness = self.f_cut
        #optimizer.params['swarm_size'] = 10
        optimizer.params['pop_init'] = 10
        optimizer.max_evaluations = self.evals
        optimizer.objectives = 1  
        #optimizer.monitoring = 'basic'
        result = optimizer.optimize()
        		
        min_f = result.f 
        min_x = result.X      
        
        predictions.append(min_x)
                    
        ind = np.argsort(solutions_f)[:self.top_rank]
        prediction_sets.append(np.array(solutions_X)[ind])
            
            
        return np.array(predictions), prediction_sets
    
    
    def inference_ensemble(self, ensemble_model_path, target):
                
        #input_data, output_data = self._load_data()
        forward_model, pca_model = self._load_models()
        
        ensemble = joblib.load(ensemble_model_path)
                
        predictions = []
        prediction_sets = []
        
        init_sets = self._get_ensemble_prediction(ensemble, pca_model.transform(target.reshape(1,-1)))
        init_sets = np.unique(init_sets, axis=0)
        

        """end of ensemble code code"""
        
        solutions_X = []
        solutions_f = []
        
        def objective_function(x):
            
            em = forward_model.predict(np.array(x).reshape(1,-1))
            em_val = pca_model.inverse_transform(em)
            rmse = np.sqrt(mean_squared_error(target, em_val))   
            
            solutions_X.append(x)
            solutions_f.append(rmse)
            
            return rmse

            
        optimizer = DE()
        
        optimizer.lb = np.array([0.2, 10, 15])
        optimizer.ub = np.array([1.3, 700, 28])
        optimizer.X0 = init_sets
        optimizer.evaluation_function = objective_function
        if self.f_cut != None:
            optimizer.target_fitness = self.f_cut
        #optimizer.params['swarm_size'] = 10
        optimizer.params['pop_init'] = 10
        optimizer.max_evaluations = self.evals
        optimizer.objectives = 1  
        #optimizer.monitoring = 'basic'
        result = optimizer.optimize()
        		
        min_f = result.f 
        min_x = result.X      
        
        predictions.append(min_x)
                    
        ind = np.argsort(solutions_f)[:self.top_rank]
        prediction_sets.append(np.array(solutions_X)[ind])
            
            
        return np.array(predictions), prediction_sets
    
    
    def inference_HF(self, target):
                
        forward_model, pca_model = self._load_models()
        
                
        predictions = []
        prediction_sets = []
        


        """end of ensemble code code"""
        
        solutions_X = []
        solutions_f = []
        
        def objective_function(x):
            
            em = forward_model.predict(np.array(x).reshape(1,-1))
            em_val = pca_model.inverse_transform(em)
            rmse = np.sqrt(mean_squared_error(target, em_val))   
            
            solutions_X.append(x)
            solutions_f.append(rmse)
            
            return rmse

            
        optimizer = DE()
        
        optimizer.lb = np.array([0.2, 10, 15])
        optimizer.ub = np.array([1.3, 700, 28])
        
        optimizer.evaluation_function = objective_function
        if self.f_cut != None:
            optimizer.target_fitness = self.f_cut
        #optimizer.params['swarm_size'] = 10
        optimizer.params['pop_init'] = 10
        optimizer.max_evaluations = self.evals
        optimizer.objectives = 1  
        #optimizer.monitoring = 'basic'
        result = optimizer.optimize()
        		
        min_f = result.f 
        min_x = result.X      
        
        predictions.append(min_x)
                    
        ind = np.argsort(solutions_f)[:self.top_rank]
        prediction_sets.append(np.array(solutions_X)[ind])
            
            
        return np.array(predictions), prediction_sets
