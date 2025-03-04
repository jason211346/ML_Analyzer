import React, { useState, useRef } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

interface FileUploaderProps {
  onFileUpload: (file: File) => void;
  isLoading: boolean;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload, isLoading }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndUploadFile(e.target.files[0]);
    }
  };

  const validateAndUploadFile = (file: File) => {
    // Check if file is an Excel or CSV file
    const validExcelTypes = [
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-excel.sheet.macroEnabled.12',
      'text/csv'
    ];
    
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    
    if (!validExcelTypes.includes(file.type) && !['xls', 'xlsx', 'csv'].includes(fileExtension || '')) {
      toast.error('請上傳有效的 Excel 或 CSV 檔案 (.xls, .xlsx 或 .csv)');
      return;
    }
    
    onFileUpload(file);
  };

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center justify-center space-y-4">
        {isLoading ? (
          <div className="animate-spin">
            <FileText size={48} className="text-blue-500" />
          </div>
        ) : (
          <>
            <Upload size={48} className="text-gray-400" />
            <h3 className="text-lg font-medium">拖曳或點擊上傳 Excel 或 CSV 檔案</h3>
            <p className="text-sm text-gray-500">支援 .xlsx、.xls 和 .csv 格式</p>
            <div className="flex items-center text-xs text-gray-500">
              <AlertCircle size={14} className="mr-1" />
              <span>檔案將儲存於本地端</span>
            </div>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500"
              disabled={isLoading}
            >
              選擇檔案
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".xlsx,.xls,.csv"
              onChange={handleFileInputChange}
              className="hidden"
              disabled={isLoading}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default FileUploader;