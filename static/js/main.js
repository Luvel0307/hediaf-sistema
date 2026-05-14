/* ============================================
   Sistema Integral de Pie Diabético - JS
   ============================================ */

// Auto-dismiss flash messages after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 5000);
    });
});

// Image preview helper
function previewImage(input, previewId) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById(previewId);
            if (preview) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// Color mapping for severity grades
const GRADE_COLORS = {
    1: '#388e3c',
    2: '#fbc02d',
    3: '#f57c00',
    4: '#d32f2f'
};

const GRADE_LABELS = {
    1: 'Normal',
    2: 'Leve',
    3: 'Moderado',
    4: 'Grave'
};

// Chart.js default config
if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
}
