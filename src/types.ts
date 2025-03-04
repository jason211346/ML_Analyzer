export interface User {
  id: string;
  name: string;
  folderPath: string;
  createdAt: string;
}

export interface ExcelData {
  headers: string[];
  data: any[][];
}

export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

export interface FeatureSelection {
  inputFeatures: string[];
  targetFeatures: string[];
}

export type ModelType = 'linear' | 'decisionTree' | 'randomForest';

export type TaskType = 'regression' | 'classification';

export interface ModelConfig {
  type: ModelType;
  name: string;
  description: string;
  hyperparameters?: Record<string, any>;
}

export interface ModelResults {
  modelType: ModelType;
  modelName: string;
  mse: number;
  r2: number;
  accuracy?: number; // For classification
  precision?: number; // For classification
  recall?: number; // For classification
  f1Score?: number; // For classification
  confusionMatrix?: number[][]; // For classification
  featureImportance: {
    feature: string;
    importance: number;
  }[];
  predictions: number[][];
  actual: number[][];
  isClassification?: boolean;
  classes?: string[] | number[]; // Class labels for classification
}

export interface TrainingData {
  X_train: number[][];
  y_train: number[][];
  X_test: number[][];
  y_test: number[][];
  featureNames: string[];
  targetNames: string[];
  isClassification?: boolean;
  classes?: string[] | number[]; // Class labels for classification
}

export interface ModelComparisonResults {
  models: ModelResults[];
  bestModel: ModelType;
}