import React, { useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import './ImageDropzone.css';


const ImageDropzone = () => {
const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
accept: { 'image/*': [] }
});


const thumbs = useMemo(() => acceptedFiles.map(file => (
<li key={file.name} className="thumb">
<img src={URL.createObjectURL(file)} alt={file.name} />
<div className="thumb-meta">
<span className="name" title={file.name}>{file.name}</span>
<span className="size">{Math.round(file.size/1024)} KB</span>
</div>
</li>
)), [acceptedFiles]);


return (
<div className="drop-wrapper">
<div {...getRootProps({ className: `dropzone ${isDragActive ? 'active' : ''}` })}>
<input {...getInputProps()} />
<div className="dz-inner">
<div className="icon">🖼️</div>
<div className="dz-text">
{isDragActive ? 'Drop your images here' : 'Drag and drop an ECG here, or click to select one'}
</div>
<div className="dz-sub">PNG, JPG, GIF – máx. 25MB</div>
</div>
</div>


{acceptedFiles.length > 0 && (
<div className="preview">
<h3>Seleccionadas</h3>
<ul className="thumbs">{thumbs}</ul>
</div>
)}
</div>
);
};


export default ImageDropzone;