import React from 'react';
import {useDropzone} from 'react-dropzone';
import './ImageDropzone.css';

const ImageDropzone = () => {
  const {getRootProps, getInputProps, isDragActive} = useDropzone({
    accept: 'image/*',
    onDrop: acceptedFiles => {
      console.log(acceptedFiles);
    }
  });

  return (
    <div {...getRootProps({ className: 'dropzone' })}>
      <input {...getInputProps()} />
      <div className="dropzone-inner">
        <svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <circle cx="8.5" cy="8.5" r="1.5"></circle>
          <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
        {
          isDragActive ?
            <p>Suelta la imagen aquí...</p> :
            <p>Arrastra y suelta una imagen aquí, o haz clic para seleccionar una</p>
        }
      </div>
    </div>
  );
};

export default ImageDropzone;
