/**
 * File system utility functions
 * 
 * Note: In a browser environment, we can't directly access the file system.
 * These functions simulate file system operations using localStorage.
 */

import { User } from '../types';

// Simulate folder creation by storing user data in localStorage
export const createUserFolder = (user: User): boolean => {
  try {
    const users = getUsersFromStorage();
    
    // Check if user already exists
    if (users.some(u => u.id === user.id)) {
      return false;
    }
    
    // Add user to storage
    users.push(user);
    localStorage.setItem('users', JSON.stringify(users));
    
    // Create empty files array for this user
    localStorage.setItem(`files_${user.id}`, JSON.stringify([]));
    
    return true;
  } catch (error) {
    console.error('Error creating user folder:', error);
    return false;
  }
};

// Simulate folder deletion by removing user data from localStorage
export const deleteUserFolder = (userId: string): boolean => {
  try {
    const users = getUsersFromStorage();
    const updatedUsers = users.filter(user => user.id !== userId);
    
    localStorage.setItem('users', JSON.stringify(updatedUsers));
    
    // Remove user's files
    localStorage.removeItem(`files_${userId}`);
    
    return true;
  } catch (error) {
    console.error('Error deleting user folder:', error);
    return false;
  }
};

// Get all users from storage
export const getUsersFromStorage = (): User[] => {
  try {
    const usersJson = localStorage.getItem('users');
    return usersJson ? JSON.parse(usersJson) : [];
  } catch (error) {
    console.error('Error getting users from storage:', error);
    return [];
  }
};

// Save file data for a specific user
export const saveFileForUser = (userId: string, fileName: string, fileData: string): boolean => {
  try {
    const filesJson = localStorage.getItem(`files_${userId}`);
    const files = filesJson ? JSON.parse(filesJson) : [];
    
    // Check if file already exists
    const existingFileIndex = files.findIndex((f: any) => f.name === fileName);
    
    if (existingFileIndex >= 0) {
      // Update existing file
      files[existingFileIndex] = {
        name: fileName,
        data: fileData,
        uploadedAt: new Date().toISOString()
      };
    } else {
      // Add new file
      files.push({
        name: fileName,
        data: fileData,
        uploadedAt: new Date().toISOString()
      });
    }
    
    localStorage.setItem(`files_${userId}`, JSON.stringify(files));
    return true;
  } catch (error) {
    console.error('Error saving file for user:', error);
    return false;
  }
};

// Delete a file for a specific user
export const deleteFileForUser = (userId: string, fileName: string): boolean => {
  try {
    const filesJson = localStorage.getItem(`files_${userId}`);
    if (!filesJson) return false;
    
    const files = JSON.parse(filesJson);
    const updatedFiles = files.filter((file: any) => file.name !== fileName);
    
    localStorage.setItem(`files_${userId}`, JSON.stringify(updatedFiles));
    return true;
  } catch (error) {
    console.error('Error deleting file for user:', error);
    return false;
  }
};

// Get all files for a specific user
export const getFilesForUser = (userId: string): any[] => {
  try {
    const filesJson = localStorage.getItem(`files_${userId}`);
    return filesJson ? JSON.parse(filesJson) : [];
  } catch (error) {
    console.error('Error getting files for user:', error);
    return [];
  }
};