window.HELP_IMPROVE_VIDEOJS = false;

function setupScrollToTop() {
  const button = document.querySelector('.scroll-to-top');
  if (!button) return;

  button.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  window.addEventListener('scroll', () => {
    button.classList.toggle('visible', window.scrollY > 300);
  });
}

function setupVideoVisibility() {
  const videos = document.querySelectorAll('.section-video');
  if (!videos.length || !('IntersectionObserver' in window)) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.play().catch(() => {});
      } else {
        entry.target.pause();
      }
    });
  }, { threshold: 0.25 });

  videos.forEach((video) => observer.observe(video));
}

document.addEventListener('DOMContentLoaded', () => {
  setupScrollToTop();
  setupVideoVisibility();
});
