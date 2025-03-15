const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Path to the SVG file
const svgFilePath = path.join(__dirname, 'book_logo.svg');
const svgBuffer = fs.readFileSync(svgFilePath);

// Convert SVG to PNG logo192.png (192x192)
sharp(svgBuffer)
  .resize(192, 192)
  .png()
  .toFile(path.join(__dirname, 'logo192.png'))
  .then(() => console.log('Created logo192.png'))
  .catch(err => console.error('Error creating logo192.png:', err));

// Convert SVG to PNG logo512.png (512x512)
sharp(svgBuffer)
  .resize(512, 512)
  .png()
  .toFile(path.join(__dirname, 'logo512.png'))
  .then(() => console.log('Created logo512.png'))
  .catch(err => console.error('Error creating logo512.png:', err));

// Convert SVG to ICO favicon.ico (32x32)
sharp(svgBuffer)
  .resize(32, 32)
  .toBuffer()
  .then(buffer => {
    // Convert PNG buffer to ICO (simply rename the extension for browser compatibility)
    fs.writeFileSync(path.join(__dirname, 'favicon.ico'), buffer);
    console.log('Created favicon.ico');
  })
  .catch(err => console.error('Error creating favicon.ico:', err));
