'use strict';

// 导入hexo输出工具类
const logger = require('hexo-log')();
const webp = require('webp-converter');
const path = require('path');
const fs = require('hexo-fs');

// 自动转码图片为webp
function convert_webp(s, title, lay, pub) {
    const reg = /!\[(.*?)\]\(\/images\/(.*?)\.(png|jpg|jpeg)\)/g;
    // $0 is all string
    // $1 is tips
    // $2 is basename
    // $3 is suffix
    const array = [...s.matchAll(reg)];
    if (array.length === 0) return s;
    for (var i of array) {
        const inputpath = path.resolve('source', 'images', i[2] + "." + i[3]);
        const outputpath = path.resolve('source', 'images', i[2] + ".webp");
        if (!fs.existsSync(outputpath)) {
            const res = webp.cwebp(inputpath, outputpath, "-q 75");
            res.then(() => {
                // 删除png文件
                fs.unlinkSync(inputpath);
                logger.info('Convert ' + inputpath+' to ' + outputpath);
            });
        }
    }
    // 替换当前内容
    var distr = "![$1](/images/$2.webp)";
    s = s.replace(reg, distr);
    // 修改文件内容
    // console.log(p, lay, pub);
    // 假定title就是文件名！
    var p = path.resolve('source');
    if (pub === false) {
        p = path.resolve(p, "_drafts", title+".md");
    } else if (pub === true) {
        p = path.resolve(p, "_posts", title+".md");
    } else {
        p = path.resolve(p, title+".md");
    }
    var fstr = fs.readFileSync(p);
    fstr = fstr.replace(reg, distr);
    fs.writeFileSync(p, fstr);
    return s;
}


hexo.extend.filter.register('before_post_render', function (data) {
    // 将\\替换为\\\\（markdown转义）
    // if (data.plugins !== undefined && data.plugins.mathjax === true) {
        // data.content = escape_double_slash(data.content);
    // }
    // 图片处理
    // data.content = convert_webp(data.content, data.title, data.layout, data.published);
    return data;
});