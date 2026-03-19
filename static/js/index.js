$(document).ready(function () {

  // ── Mobile nav burger ─────────────────────────────────────────────────
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // ── Carousel (scene images) ────────────────────────────────────────────
  var options = {
    slidesToScroll: 1,
    slidesToShow: 2,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 3000,
  };

  var carousels = bulmaCarousel.attach("#results-carousel", options);

  // ── Smooth scroll for anchor links ────────────────────────────────────
  $('a[href^="#"]').on("click", function (e) {
    var target = $(this.getAttribute("href"));
    if (target.length) {
      e.preventDefault();
      $("html, body").animate({ scrollTop: target.offset().top - 60 }, 400);
    }
  });

});
