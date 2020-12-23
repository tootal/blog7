'use strict';

module.exports = ctx => function(args, content) {
  content = ctx.render.renderSync({ text: content, engine: 'markdown' });
  return `<article class="message is-${args[0]}"><div class="message-body">${content}</div></article>`;
};