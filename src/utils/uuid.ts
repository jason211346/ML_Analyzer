/**
 * Generate a UUID v4 string
 * This is a browser-compatible implementation that doesn't rely on Node.js crypto module
 */
export function generateUUID(): string {
  // Public Domain/MIT
  let d = new Date().getTime();
  
  // Time in microseconds since page-load or 0 if unsupported
  let d2 = (performance && performance.now && (performance.now() * 1000)) || 0;
  
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    // random number between 0 and 16
    let r = Math.random() * 16;
    
    // Use timestamp + performance.now() if available for better randomness
    if (d > 0) {
      // Use timestamp bits for left side of UUID
      r = (d + r) % 16 | 0;
      d = Math.floor(d / 16);
    } else {
      // Use performance.now bits if timestamp is depleted
      r = (d2 + r) % 16 | 0;
      d2 = Math.floor(d2 / 16);
    }
    
    // Return either 'x' or 'y' character with proper hex digit
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}