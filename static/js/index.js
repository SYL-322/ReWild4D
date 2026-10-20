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

function setupGalleries() {
  document.querySelectorAll('[data-gallery]').forEach((gallery) => {
    const track = gallery.querySelector('.gallery-track');
    const slides = Array.from(track.querySelectorAll('.gallery-slide'));
    const videos = slides.map((slide) => slide.querySelector('video'));
    const count = slides.length;
    const previous = gallery.querySelector('.gallery-prev');
    const next = gallery.querySelector('.gallery-next');
    const previewPrevious = gallery.querySelector('.gallery-peek-prev');
    const previewNext = gallery.querySelector('.gallery-peek-next');
    const position = gallery.querySelector('.gallery-position');
    const dots = gallery.querySelector('.gallery-dots');
    let active = 0;
    let visible = false;
    let onBoundary = false;
    let teleporting = false;
    let scrolling = false;
    let settleTimer;
    let trackWidth = track.clientWidth;

    function wrap(index) {
      return (index + count) % count;
    }

    function boundarySlide(source) {
      const slide = document.createElement('figure');
      const image = document.createElement('img');
      const caption = document.createElement('figcaption');
      slide.className = 'gallery-slide gallery-boundary';
      slide.setAttribute('aria-hidden', 'true');
      slide.inert = true;
      image.className = 'section-video';
      image.src = source.querySelector('video').poster;
      image.alt = '';
      caption.textContent = source.querySelector('figcaption').textContent;
      slide.append(image, caption);
      return slide;
    }

    track.prepend(boundarySlide(slides[count - 1]));
    track.append(boundarySlide(slides[0]));

    function slideOffset(index) {
      return track.children[index].offsetLeft - track.firstElementChild.offsetLeft;
    }

    function nearestPhysicalIndex() {
      let nearest = 0;
      for (let index = 1; index < track.children.length; index += 1) {
        if (Math.abs(track.scrollLeft - slideOffset(index)) < Math.abs(track.scrollLeft - slideOffset(nearest))) {
          nearest = index;
        }
      }
      return nearest;
    }

    function teleportTo(physicalIndex) {
      teleporting = true;
      scrolling = false;
      track.style.scrollSnapType = 'none';
      track.scrollTo({ left: slideOffset(physicalIndex), behavior: 'auto' });
      requestAnimationFrame(() => {
        track.style.scrollSnapType = '';
        teleporting = false;
        onBoundary = false;
        syncPlayback();
      });
    }

    function syncPlayback() {
      videos.forEach((video, index) => {
        if (visible && !onBoundary && !scrolling && index === active) {
          if (video.readyState === 0) video.load();
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    }

    function updatePreview(button, index) {
      const target = wrap(index);
      const label = slides[target].querySelector('figcaption').textContent;
      button.hidden = false;
      button.title = `Show ${label}`;
      button.setAttribute('aria-label', button.title);
      button.querySelector('img').src = videos[target].poster;
    }

    function setActive(index) {
      if (index === active && dots.children[active]?.getAttribute('aria-current') === 'true') return;
      active = index;
      position.textContent = `${active + 1} / ${count}`;
      previous.disabled = false;
      next.disabled = false;
      Array.from(dots.children).forEach((dot, dotIndex) => {
        dot.setAttribute('aria-current', String(dotIndex === active));
      });
      updatePreview(previewPrevious, active - 1);
      updatePreview(previewNext, active + 1);
      syncPlayback();
    }

    function show(index) {
      const target = index === -1 ? 0 : index === count ? count + 1 : wrap(index) + 1;
      if (Math.abs(track.scrollLeft - slideOffset(target)) < 1) return;
      scrolling = true;
      syncPlayback();
      track.scrollTo({
        left: slideOffset(target),
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      });
    }

    function settle() {
      clearTimeout(settleTimer);
      const physical = nearestPhysicalIndex();
      scrolling = false;
      if (physical === 0 || physical === count + 1) {
        teleportTo(physical === 0 ? count : 1);
      } else {
        syncPlayback();
      }
    }

    slides.forEach((slide, index) => {
      const dot = document.createElement('button');
      dot.className = 'gallery-dot';
      dot.type = 'button';
      dot.title = slide.querySelector('figcaption').textContent;
      dot.setAttribute('aria-label', `Show ${dot.title}`);
      dot.addEventListener('click', () => show(index));
      dots.appendChild(dot);
    });

    previous.addEventListener('click', () => show(active - 1));
    next.addEventListener('click', () => show(active + 1));
    previewPrevious.addEventListener('click', () => show(active - 1));
    previewNext.addEventListener('click', () => show(active + 1));
    videos.forEach((video, index) => {
      video.addEventListener('canplay', () => {
        if (index === active) syncPlayback();
      });
    });
    track.addEventListener('scroll', () => {
      if (teleporting) return;
      scrolling = true;
      const physical = nearestPhysicalIndex();
      onBoundary = physical === 0 || physical === count + 1;
      setActive(wrap(physical - 1));
      syncPlayback();
      clearTimeout(settleTimer);
      settleTimer = setTimeout(settle, 180);
    }, { passive: true });
    track.addEventListener('scrollend', settle);
    track.addEventListener('keydown', (event) => {
      if (event.target !== track || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      show(active + (event.key === 'ArrowRight' ? 1 : -1));
    });

    teleportTo(1);
    setActive(0);
    function updateGeometry() {
      const width = track.clientWidth;
      gallery.style.setProperty('--gallery-video-width', `${width}px`);
      gallery.style.setProperty('--gallery-video-half-width', `${width / 2}px`);
      gallery.style.setProperty('--gallery-video-height', `${videos[0].getBoundingClientRect().height}px`);
      if (Math.abs(width - trackWidth) > 1) {
        trackWidth = width;
        teleportTo(active + 1);
      }
    }
    updateGeometry();
    if ('ResizeObserver' in window) {
      new ResizeObserver(updateGeometry).observe(track);
    }
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        syncPlayback();
      }, { threshold: 0.25 });
      observer.observe(gallery);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setupScrollToTop();
  setupGalleries();
});
