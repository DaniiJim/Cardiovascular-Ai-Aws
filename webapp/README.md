Una web interactiva sobre salud y tecnología, con carga de ECG y resultados de análisis (simulación por ahora). Construida con React + Vite.

📂 Instalación

Clona el repositorio:

git clone https://github.com/usuario/nombre-del-repo.git


Entra en la carpeta del proyecto:

cd nombre-del-repo


Instala las dependencias:

npm install

🚀 Ejecutar localmente

Para ver la web en tu navegador:

npm run dev


Luego abre la URL que muestra la terminal (normalmente http://localhost:5173).

⚙️ Estructura del proyecto

src/components/ → Componentes de React (Hero, Info, Analyze, Footer, etc.)

src/assets/ → Imágenes y recursos estáticos

src/App.jsx → Componente principal

vite.config.js → Configuración de Vite

📝 Uso

Hero: Título, subtítulo y botón de llamada a la acción.

Info: Secciones explicativas con texto e imágenes.

Analyze: Botón de carga de imagen y espacio para mostrar resultados (con API futura).

Footer: Información de contacto o links relevantes.

💡 Tecnologías

React

Vite

CSS Modules

JavaScript moderno (ES6+)

📌 Notas

La sección de Analyze todavía no tiene lógica de análisis; se puede conectar con una API más adelante.

Compatible con escritorio y móvil, diseño responsive.