$(function(){
  var $el, $ps, $up, totalHeight;

  $(".sidebar-box .btn").click(function() {

    totalHeight = 0

    $el = $(this);
    $p  = $el.parent();
    $up = $p.parent();
    $ps = $up.find("p:not('.read-more')");

    // measure how tall inside should be by adding together heights of all inside paragraphs (except read-more paragraph)
    $ps.each(function() {
      totalHeight += $(this).outerHeight();
    });

    $up
      .css({
        // Set height to prevent instant jump down when max height is removed
        "height": $up.height(),
        "max-height": 9999
      })
      .animate({
        "height": totalHeight
      })
      .click(function() {
          //After expanding, click paragraph to revert to original state
          $p.fadeIn();
          $up.animate({
              "height": 140
          });
      });

    // fade out read-more
    $p.fadeOut();

    // prevent jump-down
    return false;

  });
});