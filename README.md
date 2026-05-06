# SkinCheck AI - Skin Cancer Detection Website

A modern, multi-page React website for skin cancer detection with an AI-powered analysis tool.

![Website Preview](https://b7v4rskhdtojw.ok.kimi.link)

## Features

- **Home Page**: Hero section, features overview, how-it-works guide, and call-to-action
- **About Us**: Mission, vision, company story, and technology information
- **Contact Page**: Contact form, contact information, and FAQ section
- **Detection Tool**: Upload skin lesion images for AI-powered analysis with detailed results

## Project Structure

```
app/
├── src/
│   ├── components/          # Reusable React components
│   │   ├── Navigation.tsx   # Navigation bar with mobile menu
│   │   ├── Footer.tsx       # Footer with links and contact info
│   │   ├── UploadArea.tsx   # Image upload component with drag & drop
│   │   └── ResultDisplay.tsx # Detection results display
│   ├── pages/               # Page components
│   │   ├── Home.tsx         # Home page
│   │   ├── About.tsx        # About Us page
│   │   ├── Contact.tsx      # Contact page
│   │   └── Detection.tsx    # Skin detection tool page
│   ├── styles/              # CSS styles
│   │   └── global.css       # Global styles and variables
│   ├── App.tsx              # Main app with routing
│   └── index.css            # Tailwind + base styles
├── backend/                 # Python Flask backend
│   ├── app.py               # Flask API server
│   └── requirements.txt     # Python dependencies
├── dist/                    # Build output (generated)
└── index.html               # HTML entry point
```

## Technologies Used

### Frontend
- **React** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Utility-first CSS
- **shadcn/ui** - UI component library
- **React Router** - Client-side routing
- **Lucide React** - Icon library

### Backend
- **Flask** - Python web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Pillow** - Image processing
- **NumPy** - Numerical computing
- **TensorFlow/PyTorch** - Machine learning (add your model)

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ (for backend)

### Frontend Setup

1. Navigate to the project directory:
```bash
cd app
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open http://localhost:5173 in your browser

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your trained model:
   - Place your model file (`.h5` for TensorFlow or `.pth` for PyTorch) in the `backend/` folder
   - Update `MODEL_PATH` in `app.py` with your model's filename
   - Update the `predict_image()` function with your model's prediction code

5. Start the Flask server:
```bash
python app.py
```

The API will be available at http://localhost:5000

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Analyze skin lesion image |
| GET | `/health` | Health check |
| GET | `/classes` | Get lesion class information |

## Connecting Frontend to Backend

To connect the frontend to your backend API:

1. Update the fetch URL in `src/pages/Detection.tsx`:

```typescript
// Line ~65 in Detection.tsx
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
});
```

2. For production, update the URL to your deployed backend.

## Customization

### Colors
Edit CSS variables in `src/styles/global.css`:

```css
:root {
  --primary-color: #667eea;
  --primary-dark: #764ba2;
  /* ... */
}
```

### Content
- Update page content in respective files in `src/pages/`
- Update navigation links in `src/components/Navigation.tsx`
- Update footer content in `src/components/Footer.tsx`

### Detection Classes
Update the class definitions in `backend/app.py`:

```python
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
CLASS_EXPLANATIONS = {
    'AKIEC': 'Actinic Keratosis - Pre-cancerous lesion',
    # ...
}
```

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Deployment

### Frontend
Deploy the `dist/` folder to any static hosting service:
- Netlify
- Vercel
- GitHub Pages
- AWS S3

### Backend
Deploy the Flask app to:
- Heroku
- AWS Elastic Beanstalk
- Google Cloud Run
- PythonAnywhere

## Important Notes

### Medical Disclaimer
This tool is for **educational purposes only** and is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist for proper evaluation of skin lesions.

### Model Integration
The current implementation includes a mock prediction function. To use your actual model:

1. Update `backend/app.py` with your model loading code
2. Implement the `predict_image()` function with your model's inference
3. Adjust the `preprocess_image()` function to match your model's input requirements

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please contact:
- Email: support@skincheckai.com
- GitHub Issues: [Your Repository]

---

**Built with care for early skin cancer detection awareness.**
