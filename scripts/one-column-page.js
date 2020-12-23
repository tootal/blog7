'use strict';

hexo.extend.filter.register('before_post_render', function (data) {
    // 关闭文章目录时改为单栏
    if (data.toc === undefined || data.toc === null || data.toc === false) {
        data.widgets = null;
    }
    return data;
});