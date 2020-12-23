function onTabClick (event) {
    var target = event.currentTarget;
    var tabId = target.parentNode.parentNode.parentNode.dataset.id;
    $('.article .content .tab-content[id^="'+tabId+'"]').css('display', 'none');
    $('.article .content .tabs[data-id="'+tabId+'"] li').removeClass('is-active');
    $(document.getElementById(tabId + '/' + target.text)).css('display', 'block');
    $(target).parent().addClass('is-active');
}