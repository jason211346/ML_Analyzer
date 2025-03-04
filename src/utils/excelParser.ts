import * as XLSX from 'xlsx';
import { ExcelData } from '../types';

export const parseExcelFile = (file: File): Promise<ExcelData> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = e.target?.result;
        const workbook = XLSX.read(data, { type: 'binary' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        
        // Convert to JSON
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        
        // Extract headers (first row)
        const headers = jsonData[0] as string[];
        
        // Extract data (remaining rows)
        const rows = jsonData.slice(1) as any[][];
        
        // Validate data
        if (!headers || headers.length === 0) {
          reject(new Error('無法識別欄位名稱，請確保第一行包含欄位標題'));
          return;
        }
        
        if (rows.length === 0) {
          reject(new Error('檔案中沒有資料列'));
          return;
        }
        
        // Check for duplicate headers
        const uniqueHeaders = new Set(headers);
        if (uniqueHeaders.size !== headers.length) {
          // Find duplicates
          const seen = new Set();
          const duplicates = headers.filter(item => {
            if (seen.has(item)) return true;
            seen.add(item);
            return false;
          });
          
          reject(new Error(`檔案包含重複的欄位名稱: ${duplicates.join(', ')}`));
          return;
        }
        
        resolve({
          headers,
          data: rows
        });
      } catch (error) {
        reject(new Error('解析 Excel 檔案失敗'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('讀取 Excel 檔案失敗'));
    };
    
    reader.readAsBinaryString(file);
  });
};

// Parse CSV file
export const parseCSVFile = (file: File): Promise<ExcelData> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = e.target?.result as string;
        
        // Use XLSX to parse CSV
        const workbook = XLSX.read(data, { type: 'string' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        
        // Convert to JSON
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        
        // Extract headers (first row)
        const headers = jsonData[0] as string[];
        
        // Extract data (remaining rows)
        const rows = jsonData.slice(1) as any[][];
        
        // Validate data
        if (!headers || headers.length === 0) {
          reject(new Error('無法識別欄位名稱，請確保第一行包含欄位標題'));
          return;
        }
        
        if (rows.length === 0) {
          reject(new Error('檔案中沒有資料列'));
          return;
        }
        
        // Check for duplicate headers
        const uniqueHeaders = new Set(headers);
        if (uniqueHeaders.size !== headers.length) {
          // Find duplicates
          const seen = new Set();
          const duplicates = headers.filter(item => {
            if (seen.has(item)) return true;
            seen.add(item);
            return false;
          });
          
          reject(new Error(`檔案包含重複的欄位名稱: ${duplicates.join(', ')}`));
          return;
        }
        
        resolve({
          headers,
          data: rows
        });
      } catch (error) {
        reject(new Error('解析 CSV 檔案失敗'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('讀取 CSV 檔案失敗'));
    };
    
    reader.readAsText(file);
  });
};

// Detect file type and parse accordingly
export const parseFile = (file: File): Promise<ExcelData> => {
  const fileExtension = file.name.split('.').pop()?.toLowerCase();
  
  if (fileExtension === 'csv') {
    return parseCSVFile(file);
  } else {
    return parseExcelFile(file);
  }
};

// Convert Excel data to base64 string for storage
export const excelToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      if (e.target?.result) {
        resolve(e.target.result as string);
      } else {
        reject(new Error('Failed to convert file to base64'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsDataURL(file);
  });
};

// Parse base64 Excel data
export const parseBase64Excel = (base64Data: string): Promise<ExcelData> => {
  return new Promise((resolve, reject) => {
    try {
      // Remove data URL prefix if present
      const base64Content = base64Data.split(',')[1] || base64Data;
      
      // Convert base64 to binary string
      const binaryString = atob(base64Content);
      
      // Convert binary string to array buffer
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Parse workbook
      const workbook = XLSX.read(bytes, { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];
      
      // Convert to JSON
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
      
      // Extract headers (first row)
      const headers = jsonData[0] as string[];
      
      // Extract data (remaining rows)
      const rows = jsonData.slice(1) as any[][];
      
      // Validate data
      if (!headers || headers.length === 0) {
        reject(new Error('無法識別欄位名稱，請確保第一行包含欄位標題'));
        return;
      }
      
      if (rows.length === 0) {
        reject(new Error('檔案中沒有資料列'));
        return;
      }
      
      // Check for duplicate headers
      const uniqueHeaders = new Set(headers);
      if (uniqueHeaders.size !== headers.length) {
        // Find duplicates
        const seen = new Set();
        const duplicates = headers.filter(item => {
          if (seen.has(item)) return true;
          seen.add(item);
          return false;
        });
        
        reject(new Error(`檔案包含重複的欄位名稱: ${duplicates.join(', ')}`));
        return;
      }
      
      resolve({
        headers,
        data: rows
      });
    } catch (error) {
      reject(new Error('解析檔案資料失敗'));
    }
  });
};