import { Matrix } from 'ml-matrix';
import { ExcelData, TrainingData, ModelResults, FeatureSelection, ModelType, ModelComparisonResults, TaskType } from '../types';
import { getModelTrainer, calculateMSE, calculateR2, shuffleArray } from './mlModels';

/**
 * Splits data into training and testing sets
 * @param data The Excel data to split
 * @param featureSelection The selected features
 * @param testSize The proportion of data to use for testing (default: 0.2)
 * @param taskType Whether this is a regression or classification task
 * @returns Training and testing data
 */
export function splitData(
  data: ExcelData,
  featureSelection: FeatureSelection,
  testSize: number = 0.2,
  taskType: TaskType = 'regression'
): TrainingData {
  try {
    // Get indices of selected features
    const inputIndices = featureSelection.inputFeatures.map(feature => 
      data.headers.indexOf(feature)
    );
    
    const targetIndices = featureSelection.targetFeatures.map(feature => 
      data.headers.indexOf(feature)
    );
    
    // Check if any feature wasn't found
    if (inputIndices.some(idx => idx === -1)) {
      const missingFeatures = featureSelection.inputFeatures.filter(
        feature => data.headers.indexOf(feature) === -1
      );
      throw new Error(`找不到以下輸入特徵: ${missingFeatures.join(', ')}`);
    }
    
    if (targetIndices.some(idx => idx === -1)) {
      const missingFeatures = featureSelection.targetFeatures.filter(
        feature => data.headers.indexOf(feature) === -1
      );
      throw new Error(`找不到以下預測目標: ${missingFeatures.join(', ')}`);
    }
    
    // For classification, extract unique class labels
    let classes: (string | number)[] = [];
    if (taskType === 'classification') {
      const targetIndex = targetIndices[0]; // Use the first target for classification
      const classSet = new Set<string | number>();
      
      data.data.forEach(row => {
        if (row.length > targetIndex) {
          classSet.add(row[targetIndex]);
        }
      });
      
      classes = Array.from(classSet);
    }
    
    // Filter out rows with missing values or non-numeric values
    const validRows = data.data.filter(row => {
      // Skip rows that are too short
      if (row.length < Math.max(...inputIndices, ...targetIndices) + 1) {
        return false;
      }
      
      // Check if all selected features have valid values
      const allInputsValid = inputIndices.every(index => {
        const value = row[index];
        return value !== undefined && value !== null;
      });
      
      const allTargetsValid = targetIndices.every(index => {
        const value = row[index];
        return value !== undefined && value !== null;
      });
      
      return allInputsValid && allTargetsValid;
    });
    
    if (validRows.length === 0) {
      throw new Error('過濾後沒有有效的資料列，請檢查資料是否包含非數值或缺失值');
    }
    
    if (validRows.length < 4) {
      throw new Error(`資料列數太少 (${validRows.length})，至少需要 4 筆有效資料進行訓練和測試`);
    }
    
    // Convert string values to numbers for input features
    const processedRows = validRows.map(row => {
      const processedRow = [...row]; // Create a copy
      
      // Convert input features to numbers
      inputIndices.forEach(index => {
        const value = row[index];
        if (typeof value === 'string') {
          const numValue = Number(value);
          if (!isNaN(numValue)) {
            processedRow[index] = numValue;
          }
        }
      });
      
      // For regression, convert target features to numbers
      if (taskType === 'regression') {
        targetIndices.forEach(index => {
          const value = row[index];
          if (typeof value === 'string') {
            const numValue = Number(value);
            if (!isNaN(numValue)) {
              processedRow[index] = numValue;
            }
          }
        });
      }
      
      return processedRow;
    });
    
    // Extract X (input features) and y (target features)
    const X = processedRows.map(row => inputIndices.map(index => {
      const value = row[index];
      return typeof value === 'number' ? value : Number(value);
    }));
    
    let y: any[][];
    
    if (taskType === 'classification') {
      // For classification, convert target to class indices
      const targetIndex = targetIndices[0]; // Use first target for classification
      y = processedRows.map(row => {
        const classValue = row[targetIndex];
        const classIndex = classes.indexOf(classValue);
        return [classIndex]; // Return class index
      });
    } else {
      // For regression, extract numeric values
      y = processedRows.map(row => targetIndices.map(index => {
        const value = row[index];
        return typeof value === 'number' ? value : Number(value);
      }));
    }
    
    // Check for NaN values in input features
    const hasNaN = X.some(row => row.some(val => isNaN(val)));
    
    if (hasNaN) {
      throw new Error('輸入特徵包含非數值 (NaN)，請檢查資料格式');
    }
    
    // Shuffle and split the data
    const indices = Array.from({ length: X.length }, (_, i) => i);
    shuffleArray(indices);
    
    const splitIndex = Math.floor(X.length * (1 - testSize));
    if (splitIndex < 1) {
      throw new Error('資料量不足，無法分割為訓練集和測試集');
    }
    
    const trainIndices = indices.slice(0, splitIndex);
    const testIndices = indices.slice(splitIndex);
    
    if (trainIndices.length === 0 || testIndices.length === 0) {
      throw new Error('分割後的訓練集或測試集為空');
    }
    
    const X_train = trainIndices.map(i => X[i]);
    const y_train = trainIndices.map(i => y[i]);
    const X_test = testIndices.map(i => X[i]);
    const y_test = testIndices.map(i => y[i]);
    
    return {
      X_train,
      y_train,
      X_test,
      y_test,
      featureNames: featureSelection.inputFeatures,
      targetNames: featureSelection.targetFeatures,
      isClassification: taskType === 'classification',
      classes: taskType === 'classification' ? classes : undefined
    };
  } catch (error) {
    console.error('Error in splitData:', error);
    throw error;
  }
}

/**
 * Trains a model with the specified type
 * @param trainingData The training data
 * @param modelType The type of model to train
 * @param hyperparameters Optional hyperparameters for the model
 * @returns Model results
 */
export function trainModel(
  trainingData: TrainingData, 
  modelType: ModelType = 'linear',
  hyperparameters?: any
): ModelResults {
  try {
    const modelTrainer = getModelTrainer(modelType);
    return modelTrainer(trainingData, hyperparameters);
  } catch (error) {
    console.error(`Error training ${modelType} model:`, error);
    throw error instanceof Error 
      ? error 
      : new Error(`模型訓練失敗: 未知錯誤 (${modelType})`);
  }
}

/**
 * Trains multiple models and compares their performance
 * @param trainingData The training data
 * @param modelTypes Array of model types to train
 * @param hyperparameters Optional hyperparameters for each model type
 * @returns Comparison results
 */
export function compareModels(
  trainingData: TrainingData,
  modelTypes: ModelType[] = ['linear', 'decisionTree', 'randomForest'],
  hyperparameters?: any
): ModelComparisonResults {
  try {
    const results: ModelResults[] = [];
    
    // Train each model type
    for (const modelType of modelTypes) {
      try {
        const modelResult = trainModel(trainingData, modelType, hyperparameters?.[modelType]);
        results.push(modelResult);
      } catch (error) {
        console.error(`Error training ${modelType} model:`, error);
        // Continue with other models
      }
    }
    
    if (results.length === 0) {
      throw new Error('所有模型訓練都失敗了，請檢查資料');
    }
    
    // Find the best model based on appropriate metric
    let bestModel: ModelResults;
    
    if (trainingData.isClassification) {
      // For classification, use accuracy
      bestModel = results.reduce((best, current) => 
        (current.accuracy || 0) > (best.accuracy || 0) ? current : best
      );
    } else {
      // For regression, use R2 score
      bestModel = results.reduce((best, current) => 
        current.r2 > best.r2 ? current : best
      );
    }
    
    return {
      models: results,
      bestModel: bestModel.modelType
    };
  } catch (error) {
    console.error('Error comparing models:', error);
    throw error instanceof Error 
      ? error 
      : new Error('模型比較失敗: 未知錯誤');
  }
}

/**
 * Performs k-fold cross-validation
 * @param data The Excel data
 * @param featureSelection The selected features
 * @param modelType The type of model to validate
 * @param k Number of folds
 * @param taskType Whether this is a regression or classification task
 * @param hyperparameters Optional hyperparameters for the model
 * @returns Average metrics across folds
 */
export function crossValidate(
  data: ExcelData,
  featureSelection: FeatureSelection,
  modelType: ModelType = 'linear',
  k: number = 5,
  taskType: TaskType = 'regression',
  hyperparameters?: any
): { avgMSE: number; avgR2: number; avgAccuracy?: number } {
  try {
    // Get indices of selected features
    const inputIndices = featureSelection.inputFeatures.map(feature => 
      data.headers.indexOf(feature)
    );
    
    const targetIndices = featureSelection.targetFeatures.map(feature => 
      data.headers.indexOf(feature)
    );
    
    // Check if any feature wasn't found
    if (inputIndices.some(idx => idx === -1)) {
      const missingFeatures = featureSelection.inputFeatures.filter(
        feature => data.headers.indexOf(feature) === -1
      );
      throw new Error(`找不到以下輸入特徵: ${missingFeatures.join(', ')}`);
    }
    
    if (targetIndices.some(idx => idx === -1)) {
      const missingFeatures = featureSelection.targetFeatures.filter(
        feature => data.headers.indexOf(feature) === -1
      );
      throw new Error(`找不到以下預測目標: ${missingFeatures.join(', ')}`);
    }
    
    // Filter out rows with missing values or non-numeric values
    const validRows = data.data.filter(row => {
      // Skip rows that are too short
      if (row.length < Math.max(...inputIndices, ...targetIndices) + 1) {
        return false;
      }
      
      const allInputsValid = inputIndices.every(index => {
        const value = row[index];
        return value !== undefined && value !== null;
      });
      
      const allTargetsValid = targetIndices.every(index => {
        const value = row[index];
        return value !== undefined && value !== null;
      });
      
      return allInputsValid && allTargetsValid;
    });
    
    if (validRows.length === 0) {
      throw new Error('過濾後沒有有效的資料列，請檢查資料是否包含非數值或缺失值');
    }
    
    // Adjust k if there are too few samples
    const adjustedK = Math.min(k, Math.floor(validRows.length / 2));
    if (adjustedK < 2) {
      throw new Error(`資料列數太少 (${validRows.length})，無法進行交叉驗證`);
    }
    
    // For classification, extract unique class labels
    let classes: (string | number)[] = [];
    if (taskType === 'classification') {
      const targetIndex = targetIndices[0]; // Use the first target for classification
      const classSet = new Set<string | number>();
      validRows.forEach(row => {
        classSet.add(row[targetIndex]);
      });
      classes = Array.from(classSet);
    }
    
    // Convert string values to numbers for input features
    const processedRows = validRows.map(row => {
      const processedRow = [...row]; // Create a copy
      
      // Convert input features to numbers
      inputIndices.forEach(index => {
        const value = row[index];
        if (typeof value === 'string') {
          const numValue = Number(value);
          if (!isNaN(numValue)) {
            processedRow[index] = numValue;
          }
        }
      });
      
      // For regression, convert target features to numbers
      if (taskType === 'regression') {
        targetIndices.forEach(index => {
          const value = row[index];
          if (typeof value === 'string') {
            const numValue = Number(value);
            if (!isNaN(numValue)) {
              processedRow[index] = numValue;
            }
          }
        });
      }
      
      return processedRow;
    });
    
    // Extract X (input features) and y (target features)
    const X = processedRows.map(row => inputIndices.map(index => {
      const value = row[index];
      return typeof value === 'number' ? value : Number(value);
    }));
    
    let y: any[][];
    
    if (taskType === 'classification') {
      // For classification, convert target to class indices
      const targetIndex = targetIndices[0]; // Use first target for classification
      y = processedRows.map(row => {
        const classValue = row[targetIndex];
        const classIndex = classes.indexOf(classValue);
        return [classIndex]; // Return class index
      });
    } else {
      // For regression, extract numeric values
      y = processedRows.map(row => targetIndices.map(index => {
        const value = row[index];
        return typeof value === 'number' ? value : Number(value);
      }));
    }
    
    // Check for NaN values in input features
    const hasNaN = X.some(row => row.some(val => isNaN(val)));
    
    if (hasNaN) {
      throw new Error('輸入特徵包含非數值 (NaN)，請檢查資料格式');
    }
    
    // Shuffle the data
    const indices = Array.from({ length: X.length }, (_, i) => i);
    shuffleArray(indices);
    
    const shuffledX = indices.map(i => X[i]);
    const shuffledY = indices.map(i => y[i]);
    
    // Split into k folds
    const foldSize = Math.floor(shuffledX.length / adjustedK);
    const folds = Array.from({ length: adjustedK }, (_, i) => {
      const start = i * foldSize;
      const end = i === adjustedK - 1 ? shuffledX.length : start + foldSize;
      return {
        indices: indices.slice(start, end)
      };
    });
    
    // Get the model trainer function
    const modelTrainer = getModelTrainer(modelType);
    
    // Perform k-fold cross-validation
    let totalMSE = 0;
    let totalR2 = 0;
    let totalAccuracy = 0;
    let successfulFolds = 0;
    
    for (let i = 0; i < adjustedK; i++) {
      try {
        // Create test set from current fold
        const testIndices = folds[i].indices;
        
        // Skip if test set is empty
        if (testIndices.length === 0) {
          console.warn(`Fold ${i} has empty test set, skipping`);
          continue;
        }
        
        const X_test = testIndices.map(idx => shuffledX[idx]);
        const y_test = testIndices.map(idx => shuffledY[idx]);
        
        // Create training set from all other folds
        const trainIndices = [];
        for (let j = 0; j < adjustedK; j++) {
          if (j !== i) {
            trainIndices.push(...folds[j].indices);
          }
        }
        
        // Skip if training set is empty
        if (trainIndices.length === 0) {
          console.warn(`Fold ${i} has empty training set, skipping`);
          continue;
        }
        
        const X_train = trainIndices.map(idx => shuffledX[idx]);
        const y_train = trainIndices.map(idx => shuffledY[idx]);
        
        // Create training data object
        const trainingData = {
          X_train,
          y_train,
          X_test,
          y_test,
          featureNames: featureSelection.inputFeatures,
          targetNames: featureSelection.targetFeatures,
          isClassification: taskType === 'classification',
          classes: taskType === 'classification' ? classes : undefined
        };
        
        // Train and evaluate model
        const modelResults = modelTrainer(trainingData, hyperparameters);
        
        if (!isNaN(modelResults.mse) && isFinite(modelResults.mse) && 
            !isNaN(modelResults.r2) && isFinite(modelResults.r2)) {
          totalMSE += modelResults.mse;
          totalR2 += modelResults.r2;
          
          if (taskType === 'classification' && modelResults.accuracy !== undefined) {
            totalAccuracy += modelResults.accuracy;
          }
          
          successfulFolds++;
        }
      } catch (error) {
        console.error(`Error in fold ${i}:`, error);
        // Continue with other folds
      }
    }
    
    if (successfulFolds === 0) {
      // Return default values instead of throwing an error
      console.warn('No successful folds in cross-validation, returning default values');
      return taskType === 'classification' 
        ? { avgMSE: Infinity, avgR2: -Infinity, avgAccuracy: 0 }
        : { avgMSE: Infinity, avgR2: -Infinity };
    }
    
    const result = {
      avgMSE: totalMSE / successfulFolds,
      avgR2: totalR2 / successfulFolds
    };
    
    if (taskType === 'classification') {
      return {
        ...result,
        avgAccuracy: totalAccuracy / successfulFolds
      };
    }
    
    return result;
  } catch (error) {
    console.error('Error in cross-validation:', error);
    // Return default values instead of throwing an error
    return taskType === 'classification'
      ? { avgMSE: 0, avgR2: 0, avgAccuracy: 0 }
      : { avgMSE: 0, avgR2: 0 };
  }
}