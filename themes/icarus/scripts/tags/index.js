/* global hexo */

'use strict';

const postTabs = require('./tabs')(hexo);

hexo.extend.tag.register('tabs', postTabs, true);


const postMsg = require('./msg')(hexo);

hexo.extend.tag.register('msg', postMsg, true);
