/* eslint no-extend-native: ["error", { "exceptions": ["String"] }] */
/* global getGETQueryValue */
/* global getGETQueriesMatchingRegularExpression */
/* global setup */
/* global showdown */
/* global hljs */
/* global webots */
/* exported openTabFromEvent */

'use strict';

var handle;

if (typeof String.prototype.startsWith !== 'function') {
  String.prototype.startsWith = function(prefix) {
    return this.slice(0, prefix.length) === prefix;
  };
}

if (typeof String.prototype.endsWith !== 'function') {
  String.prototype.endsWith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
  };
}

function isInternetExplorer() {
  var userAgent = navigator.userAgent;
  return userAgent.indexOf('MSIE') !== -1 || userAgent.indexOf('Trident') !== -1;
};

var localSetup = (typeof setup === 'undefined') ? {} : setup;
var isOpenDRUrl = location.href.indexOf('opendr.eu/doc') !== -1;

function setupOpenDRUrl(url) {
  localSetup.book = 'reference';
  localSetup.page = 'index';
  localSetup.anchor = '';
  if (!localSetup.tabs)
    localSetup.tabs = {};

  var m = url.match(new RegExp('/([^/]+)/([^/\\?#]+)([^/]*)$'));
  if (m) {
    localSetup.book = m[1];
    localSetup.page = m[2];
    var args = m[3];

    m = url.match(/version=([^&#]*)/);
    if (m) {
      var version = m[1];
      var n = version.indexOf(':');
      //if (n === -1)
      //  localSetup.branch = version;
      //else
      //  localSetup.branch = version.substr(n + 1);
    }

    // Extract tab options
    var tabRegex = /[?&](tab-[^=]+)=([^&#]+)/g;
    while ((m = tabRegex.exec(url)) !== null)
      localSetup.tabs[m[1]] = m[2];

    m = args.match(/#([^&#]*)/);
    if (m)
      localSetup.anchor = m[1];
    else
      localSetup.anchor = '';
  }
}

function setupDefaultUrl(url) {
  var m;

  m = url.match(/page=([^&#?]*)/);
  if (m)
    localSetup.page = m[1].replace(/.md$/, '');
  else
    localSetup.page = 'index';

  m = url.match(/book=([^&#?]*)/);
  if (m)
    localSetup.book = m[1];
  else if (!localSetup.book)
    localSetup.book = 'reference';

  // Extract tab options
  if (!localSetup.tabs)
    localSetup.tabs = {};
  var tabRegex = /[?&](tab-[^=]+)=([^&#]+)/g;
  while ((m = tabRegex.exec(url)) !== null)
    localSetup.tabs[m[1]] = m[2];

  m = url.match(/#([^&#]*)/);
  if (m)
    localSetup.anchor = m[1];
  else
    localSetup.anchor = '';
}

function setupUrl(url) {
  if (isOpenDRUrl)
    setupOpenDRUrl(url);
  else
    setupDefaultUrl(url);

  var tabsQuery = '';
  for (var option in localSetup.tabs) {
    if (!localSetup.tabs[option])
      continue;
    if (tabsQuery)
      tabsQuery += ',';
    tabsQuery += option + '=' + localSetup.tabs[option];
  }
  tabsQuery = '[' + tabsQuery + ']';
  console.log('book=' + localSetup.book + ' page=' + localSetup.page + ' branch=' + localSetup.branch + ' tabs=' + tabsQuery + ' anchor=' + localSetup.anchor);
}

function computeTargetPath() {
  var branch = '';
  var targetPath = '';
  //if (localSetup.branch)
  //  branch = localSetup.branch;
  if (localSetup.url.startsWith('http'))
    targetPath = localSetup.url + 'docs/';
  targetPath += localSetup.book + '/';
  return targetPath;
}

function redirectUrls(node) {
  // redirect a's href
  var as = node.querySelectorAll('a');
  for (var i = 0; i < as.length; i++) {
    var a = as[i];
    var href = a.getAttribute('href');
    if (!href)
      continue;
    else if (href.startsWith('#'))
      addDynamicAnchorEvent(a); // on firefox, the second click on the anchor is not dealt cleanly
    else if (href.startsWith('http')) // open external links in a new window
      a.setAttribute('target', '_blank');
    else if (href.endsWith('.md') || href.indexOf('.md#') > -1) {
      var match, newPage, anchor;
      if (href.startsWith('../')) { // Cross-book hyperlink case.
        match = /^..\/([\w-]+)\/([\w-]+).md(#[\w-]+)?$/.exec(href);
        if (match && match.length >= 3) {
          var book = match[1];
          newPage = match[2];
          anchor = match[3];
          if (anchor)
            anchor = anchor.substring(1); // remove the '#' character
          a.setAttribute('href', forgeUrl(book, newPage, localSetup.tabs, anchor));
        }
      } else { // Cross-page hyperlink case.
        addDynamicLoadEvent(a);
        match = /^([\w-]+).md(#[\w-]+)?$/.exec(href);
        if (match && match.length >= 2) {
          newPage = match[1];
          anchor = match[2];
          if (anchor)
            anchor = anchor.substring(1); // remove the '#' character
          a.setAttribute('href', forgeUrl(localSetup.book, newPage, localSetup.tabs, anchor));
        }
      }
    }
  }
}

function collapseMovies(node) {
  if (location.href.startsWith('file:')) { // if it's the offline documentation embedded in Webots (possibly without network):
    var iframes = node.querySelectorAll('iframe');
    for (var i = 0; i < iframes.length; i++) { // foreach iframe:
      var iframe = iframes[i];
      var src = iframe.getAttribute('src');
      if (src && src.indexOf('youtube')) { // if the iframe is a youtube frame:
        // then, replace the iframe by a text and an hyperlink to the youtube page.
        src = src.replace(/embed\/(.*)\?rel=0/, 'watch?v=$1'); // e.g. https://www.youtube.com/embed/vFwNwT8dZTU?rel=0 to https://www.youtube.com/watch?v=vFwNwT8dZTU
        var p = document.createElement('p');
        p.innerHTML = '<a href="' + src + '">Click here to see the youtube movie.</a>';
        iframe.parentNode.replaceChild(p, iframe);
      }
    }
  }
}

function forgeUrl(book, page, tabs, anchor) {
  var tabOption;
  var isFirstArgument;
  var anchorString = (anchor && anchor.length > 0) ? ('#' + anchor) : '';
  var url = location.href;
  if (isOpenDRUrl) {
    var i = location.href.indexOf('opendr.eu/doc');
    url = location.href.substr(0, i) + 'opendr.eu/doc/' + book + '/' + page;
    if (localSetup.branch !== '' && localSetup.repository && localSetup.repository !== 'opendr')
      url += '?version=' + localSetup.repository + ':' + localSetup.branch;
    else if (localSetup.branch !== '')
      url += '?version=' + localSetup.branch;
    isFirstArgument = localSetup.branch === '';
    for (tabOption in tabs) {
      if (!tabs[tabOption])
        continue;
      url += (isFirstArgument ? '?' : '&') + tabOption + '=' + tabs[tabOption];
      isFirstArgument = false;
    }
    url += anchorString;
  } else {
    isFirstArgument = (url.indexOf('?') < 0);

    // Remove anchor from url
    url = url.split('#')[0];

    // Add or replace the book argument.
    if (url.indexOf('book=') > -1)
      url = url.replace(/book=([^&]+)?/, 'book=' + book);
    else {
      url += (isFirstArgument ? '?' : '&') + 'book=' + book;
      isFirstArgument = false;
    }

    // Add or replace the page argument.
    if (url.indexOf('page=') > -1)
      url = url.replace(/page=([\w-]+)?/, 'page=' + page);
    else {
      url += (isFirstArgument ? '?' : '&') + 'page=' + page;
      isFirstArgument = false;
    }

    // Add or replace the tab argument.
    for (tabOption in tabs) {
      let tabName = tabs[tabOption] ? tabs[tabOption] : '';
      if (url.indexOf(tabOption + '=') > -1)
        url = url.replace(new RegExp(tabOption + '=([^&]+)(#[\\w-]+)?'), tabOption + '=' + tabName);
      else if (tabName)
        url += (isFirstArgument ? '?' : '&') + tabOption + '=' + tabName;
    }

    url += anchorString;
  }
  return url;
}

function addDynamicAnchorEvent(el) {
  if (el.classList.contains('dynamicAnchor'))
    return;
  el.addEventListener('click',
    function(event) {
      if (event.ctrlKey)
        return;
      var node = event.target;
      while (node && !node.hasAttribute('href'))
        node = node.getParent();
      if (node) {
        localSetup.anchor = extractAnchor(node.getAttribute('href'));
        applyAnchor();
        event.preventDefault();
      }
    },
    false
  );
  el.classList.add('dynamicAnchor');
}

function addDynamicLoadEvent(el) {
  if (el.classList.contains('dynamicLoad'))
    return;
  el.addEventListener('click',
    function(event) {
      if (event.ctrlKey)
        return;
      aClick(event.target);
      event.preventDefault();
    },
    false
  );
  el.classList.add('dynamicLoad');
}

function aClick(el) {
  setupUrl(el.getAttribute('href'));
  getMDFile();
  updateBrowserUrl();
}

function redirectImages(node) {
  // redirect img's src
  var imgs = node.querySelectorAll('img');
  var targetPath = computeTargetPath();
  for (var i = 0; i < imgs.length; i++) {
    var img = imgs[i];
    var src = img.getAttribute('src');
    var match = /^images\/(.*)$/.exec(src);
    if (match && match.length === 2)
      img.setAttribute('src', targetPath + 'images/' + match[1]);
  }
}

function setupModalWindow() {
  var doc = document.querySelector('#opendr-doc');

  // Create the following HTML tags:
  // <div id="modal-window" class="modal-window">
  //   <span class="modal-window-close-button">&times;</span>
  //   <img class="modal-window-image-content" />
  //   <div class="modal-window-caption"></div>
  // </div>

  var close = document.createElement('span');
  close.classList.add('modal-window-close-button');
  close.innerHTML = '&times;';
  close.onclick = function() {
    modal.style.display = 'none';
  };

  var image = document.createElement('img');
  image.classList.add('modal-window-image-content');

  var caption = document.createElement('div');
  caption.classList.add('modal-window-caption');

  var modal = document.createElement('div');
  modal.setAttribute('id', 'modal-window');
  modal.classList.add('modal-window');

  modal.appendChild(close);
  modal.appendChild(image);
  modal.appendChild(caption);
  doc.appendChild(modal);

  window.onclick = function(event) {
    if (event.target === modal) {
      modal.style.display = 'none';
      image.style.display = 'none';
    }
  };
}

function updateModalEvents(view) {
  var modal = document.querySelector('#modal-window');
  var image = modal.querySelector('.modal-window-image-content');
  var caption = modal.querySelector('.modal-window-caption');

  // Add the modal events on each image.
  var imgs = view.querySelectorAll('img');
  for (var i = 0; i < imgs.length; i++) {
    imgs[i].onclick = function(event) {
      var img = event.target;
      // The modal window is only enabled on big enough images and on thumbnail.
      if (img.src.indexOf('thumbnail') === -1 && !(img.naturalWidth > 128 && img.naturalHeight > 128))
        return;

      // Show the modal window and the caption.
      modal.style.display = 'block';
      caption.innerHTML = (typeof this.parentNode.childNodes[1] !== 'undefined') ? this.parentNode.childNodes[1].innerHTML : '';

      if (img.src.indexOf('.thumbnail.') === -1) {
        // this is not a thumbnail => show the image directly.
        image.src = img.src;
        image.style.display = 'block';
      } else {
        // this is a thumbnail => load the actual image.
        var url = img.src.replace('.thumbnail.', '.');
        if (image.src === url) {
          // The image has already been loaded.
          image.style.display = 'block';
          return;
        } else {
          // The image has to be loaded: show the loading image.
          image.style.display = 'none';
        }
        // In case of thumbnail, search for the original png or jpg
        image.onload = function() {
          // The original image has been loaded successfully => show it.
          image.style.display = 'block';
        };
        image.onerror = function() {
          // The original image has not been loaded successfully => try to change the extension and reload it.
          image.onerror = function() {
            // The original image has not been loaded successfully => abort.
            modal.style.display = 'none';
            image.style.display = 'none';
          };
          url = img.src.replace('.thumbnail.jpg', '.png');
          image.src = url;
        };
        image.src = url;
      }
    };
  }
}

function applyAnchor() {
  var firstAnchor = document.querySelector("[name='" + localSetup.anchor + "']");
  if (firstAnchor)
    firstAnchor.scrollIntoView(true);
  else
    window.scrollTo(0, 0);
}

function applyToTitleDiv() {
  var titleContentElement = document.querySelector('#title-content');
  if (titleContentElement) {
    var newTitle;
    if (localSetup.book === 'reference')
      newTitle = 'OpenDR Reference Manual';
    else
      newTitle = '';
    if (newTitle.length > 0) {
      newTitle += " <div class='release-tag'>" + getOpenDRVersion() + '</div>';
      titleContentElement.innerHTML = newTitle;
    }
  }
}

function createIndex(view) {
  // Note: the previous index is cleaned up when the parent title is destroyed.

  // Get all the view headings.
  var headings = [].slice.call(view.querySelectorAll('h1, h2, h3, h4'));

  // Do not create too small indexes.
  var content = document.querySelector('#content');
  if ((content.offsetHeight < 2 * window.innerHeight || headings.length < 4) && (localSetup.book !== 'discord' || headings.length < 2))
    return;

  var level = parseInt(headings[0].tagName[1]) + 1; // current heading level.

  // Create an empty index, and insert it before the second heading.
  var indexTitle = document.createElement('h' + level);
  indexTitle.textContent = 'Index';
  indexTitle.setAttribute('id', 'indexTitle');
  headings[0].parentNode.insertBefore(indexTitle, headings[1]);
  var ul = document.createElement('ul');
  ul.setAttribute('id', 'index');
  headings[0].parentNode.insertBefore(ul, headings[1]);

  headings.forEach(function(heading, i) {
    if (i === 0) // Skip the first heading.
      return;

    // Update current level and ul.
    var newLevel = parseInt(heading.tagName[1]);
    while (newLevel > level) {
      var newUl = document.createElement('ul');
      ul.appendChild(newUl);
      ul = newUl;
      level += 1;
    }
    while (newLevel < level) {
      ul = ul.parentNode;
      level -= 1;
    }

    // Add the <li> tag.
    var anchor = heading.getAttribute('name');
    var a = document.createElement('a');
    a.setAttribute('href', '#' + anchor);
    a.textContent = heading.textContent;
    var li = document.createElement('li');
    li.appendChild(a);
    ul.appendChild(li);
  });
}

function getOpenDRVersion() {
  if (localSetup.branch)
    return localSetup.branch;
  // Get the OpenDR toolkit version from the showdown wbVariables extension
  var version = '{{ opendr.version }}';
  var converter = new showdown.Converter({extensions: ['wbVariables']});
  var html = converter.makeHtml(version);
  var tmp = document.createElement('div');
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || '';
}

function applyToPageTitle(mdContent) {
  var hashtagIndex = mdContent.indexOf('#');
  if (hashtagIndex >= 0) {
    while (hashtagIndex + 1 < mdContent.length && mdContent[hashtagIndex + 1] === '#')
      hashtagIndex += 1;
    var hashtagCarriageReturn = mdContent.indexOf('\n', hashtagIndex);
    if (hashtagCarriageReturn >= 0) {
      var title = mdContent.substring(hashtagIndex + 1, hashtagCarriageReturn).trim();
      document.title = 'OpenDR documentation: ' + title;
    }
  }
}

function populateViewDiv(mdContent) {
  setupUrl(document.location.href);

  var view = document.querySelector('#view');
  while (view.firstChild)
    view.removeChild(view.firstChild);

  // console.log('Raw MD content:\n\n');
  // console.log(mdContent);

  applyToPageTitle(mdContent);

  // markdown to html
  //window.mermaidGraphCounter = 0;
  //window.mermaidGraphs = {};
  var converter = new showdown.Converter({tables: 'True', extensions: ['wbTabComponent', 'wbSpoiler', 'wbChart', 'wbVariables', 'wbAPI', 'wbFigure', 'wbAnchors', 'wbIllustratedSection', 'youtube']});
  var html = converter.makeHtml(mdContent);

  // console.log('HTML content: \n\n')
  // console.log(html);

  view.innerHTML = html;

  renderGraphs();
  redirectImages(view);
  updateModalEvents(view);
  redirectUrls(view);
  collapseMovies(view);

  applyAnchorIcons(view);
  highlightCode(view);

  updateSelection();
  createIndex(view);

  var images = view.querySelectorAll('img');
  if (images.length > 0) {
    // apply the anchor only when the images are loaded,
    // otherwise, the anchor can be overestimated.
    var lastImage = images[images.length - 1];
    $(lastImage).load(applyAnchor);
  } else
    applyAnchor();
  applyTabs();
}

// replace the browser URL after a dynamic load
function updateBrowserUrl() {
  var url = forgeUrl(localSetup.book, localSetup.page, localSetup.tabs, localSetup.anchor);
  if (history.pushState) {
    try {
      history.pushState({state: 'new'}, null, url);
    } catch (err) {
    }
  }
  var canonicalUrl = 'http://localhost:8000/docs/' + localSetup.book + '/' + localSetup.page;
  $('link[rel="canonical"]').attr('href', canonicalUrl);
}

// Make in order that the back button is working correctly
window.onpopstate = function(event) {
  setupUrl(document.location.href);
  getMDFile();
};

function highlightCode(view) {
  var supportedLanguages = ['c', 'cpp', 'java', 'python', 'matlab', 'sh', 'ini', 'tex', 'makefile', 'lua', 'xml'];

  for (var i = 0; i < supportedLanguages.length; i++) {
    var language = supportedLanguages[i];
    hljs.configure({languages: [ language ]});
    var codes = document.querySelectorAll('.' + language);
    for (var j = 0; j < codes.length; j++) {
      var code = codes[j];
      hljs.highlightBlock(code);
    }
  }
}

// Open a tab component tab
function openTabFromEvent(evt, option, name) {
  // update links
  var a = document.querySelectorAll('a');
  for (var i = 0; i < a.length; i++) {
    var href = a[i].getAttribute('href');
    if (!href)
      continue;
    if (localSetup.tabs[option]) {
      if (href.includes(option + '=' + localSetup.tabs[option]))
        a[i].setAttribute('href', href.replace(option + '=' + localSetup.tabs[option], option + '=' + name.toLowerCase()));
      else if (!href.startsWith('#'))
        a[i].setAttribute('href', href + (href.indexOf('?') > -1 ? '&' : '?') + option + '=' + name.toLowerCase());
    }
  }
  // open tab
  localSetup.tabs[option] = name.toLowerCase();
  updateBrowserUrl();
  openTab(evt.target.parentNode, localSetup.tabs[option]);
}

// Open a tab component tab
function openTab(tabcomponent, name) {
  var tabID = tabcomponent.getAttribute('tabid');

  var tabcontent = tabcomponent.parentNode.querySelectorAll('.tab-content[tabid="' + tabID + '"][name="' + name + '"]')[0];
  if (typeof tabcontent === 'undefined')
    return false;

  var tabcontents = tabcomponent.parentNode.querySelectorAll('.tab-content[tabid="' + tabID + '"]');
  for (var i = 0; i < tabcontents.length; i++)
    tabcontents[i].style.display = 'none';

  var tablinks = tabcomponent.querySelectorAll('.tab-links');
  for (var j = 0; j < tablinks.length; j++)
    tablinks[j].classList.remove('active');

  tabcontent = tabcomponent.parentNode.querySelectorAll('.tab-content[tabid="' + tabID + '"][name="' + name + '"]')[0];
  tabcontent.style.display = 'block';

  var tablink = tabcomponent.querySelectorAll('.tab-links[name="' + name + '"]')[0];
  tablink.classList.add('active');
  return true;
}

function applyTabs() {
  var tabComponents = document.querySelectorAll('.tab-component');
  for (var k = 0; k < tabComponents.length; k++) {
    for (var tabName in localSetup.tabs) {
      if (openTab(tabComponents[k], localSetup.tabs[tabName]))
        break;
    }
  }
}

function renderGraphs() {
  /*for (var id in window.mermaidGraphs) {
    window.mermaidAPI.render(id, window.mermaidGraphs[id], function(svgCode, bindFunctions) {
      document.querySelector('#' + id + 'Div').innerHTML = svgCode;
      // set min-width to be 2/3 of the max-width otherwise the text might become too small
      var element = document.querySelector('#' + id);
      var style = element.getAttribute('style');
      element.setAttribute('style', style + ' min-width:' + Math.floor(0.66 * parseInt(style.split('max-width:')[1].split('px'))) + 'px;');
    });
  }*/
}

function applyAnchorIcons(view) {
  var elements = [];
  var tags = ['figcaption', 'h1', 'h2', 'h3', 'h4', 'h5'];
  var i;
  for (i = 0; i < tags.length; i++) {
    var array = Array.prototype.slice.call(view.querySelectorAll(tags[i]));
    elements = elements.concat(array);
  }
  for (i = 0; i < elements.length; i++) {
    var el = elements[i];
    var name = null;
    if (el.parentNode && el.tagName.toLowerCase() === 'figcaption' && el.parentNode.tagName.toLowerCase() === 'figure')
      name = el.parentNode.getAttribute('name');
    else
      name = el.getAttribute('name');
    if (name) {
      el.classList.add('anchor-header');
      var span = document.createElement('span');
      span.classList.add('anchor-link-image');
      var a = document.createElement('a');
      a.setAttribute('href', '#' + name);
      a.classList.add('anchor-link');
      a.appendChild(span);
      el.insertBefore(a, el.firstChild);
    }
  }
}

function receiveMenuContent(menuContent) {
  // console.log('Menu content:\n\n');
  // console.log(menuContent);

  var menu = null;

  var converter = new showdown.Converter();
  var html = converter.makeHtml(menuContent);
  var div = document.createElement('div');
  div.innerHTML = html;

  for (var i = 0; i < div.childNodes.length; i++) {
    var child = div.childNodes[i];
    if (child && child.tagName && child.tagName.length > 0 && child.tagName.toLowerCase() === 'ul') {
      menu = child;
      break;
    }
  }

  if (!menu) {
    console.error('Cannot extract Menu.');
    return;
  }

  populateMenu(menu);
  redirectUrls(menu);
  updateSelection();
}

function updateMenuScrollbar() {
  var e = document.documentElement;
  var t = document.documentElement.scrollTop || document.body.scrollTop;
  var p = e.scrollHeight - t - e.clientHeight;
  var footerHeight = 192;
  if (p < footerHeight)
    document.querySelector('#left').style.height = (e.clientHeight - footerHeight + p) + 'px';
  else
    document.querySelector('#left').style.height = '100%';
}

function updateSelection() {
  var selected = changeMenuSelection();
  populateNavigation(selected);
  if (isOpenDRUrl)
    updateMenuScrollbar();
}

function changeMenuSelection() {
  var menu = document.querySelector('#menu');
  var selecteds = [].slice.call(menu.querySelectorAll('.selected'));
  var i;
  var selected;
  for (i = 0; i < selecteds.length; i++) {
    selected = selecteds[i];
    selected.classList.remove('selected');
  }
  var as = menu.querySelectorAll('a');
  for (i = 0; i < as.length; i++) {
    var a = as[i];
    var href = a.getAttribute('href');
    var selection;
    if (!isOpenDRUrl) {
      var pageIndex = href.indexOf('page=' + localSetup.page);
      // Notes:
      // - the string length test is done to avoid wrong positive cases
      //   where a page is a prefix of another.
      // - 5 matches with the 'page=' string length.
      if (pageIndex > -1 && (5 + pageIndex + localSetup.page.length) === href.length)
        selection = true;
      else
        selection = false;
    } else {
      var n = href.indexOf('?');
      if (n > -1)
        href = href.substring(0, n);
      n = href.indexOf('#');
      if (n > -1)
        href = href.substring(0, n);
      if (href.endsWith('/doc/' + localSetup.book + '/' + localSetup.page))
        selection = true;
      else
        selection = false;
    }
    if (selection) {
      selected = a.parentNode;
      selected.classList.add('selected');
      if (selected.parentNode.parentNode.tagName.toLowerCase() === 'li') {
        selected.parentNode.parentNode.classList.add('selected');
        var firstChild = selected.parentNode.parentNode.firstChild;
        if (firstChild.tagName.toLowerCase() === 'a')
          showAccodionItem(firstChild);
      } else
        showAccodionItem(a);
      return selected;
    }
  }
}

function populateNavigation(selected) {
  var next = document.querySelector('#next');
  var previous = document.querySelector('#previous');
  var up = document.querySelector('#up');
  var toc = document.querySelector('#toc');
  var as;

  toc.setAttribute('href', forgeUrl(localSetup.book, 'menu'));
  addDynamicLoadEvent(toc);

  if (!selected) {
    next.classList.add('disabled');
    previous.classList.add('disabled');
    up.classList.add('disabled');
    return;
  }

  if (next) {
    var nextElement = null;

    var nextLiSibling = selected.nextSibling;
    while (nextLiSibling) {
      if (nextLiSibling.tagName && nextLiSibling.tagName.toLowerCase() === 'li')
        break;
      nextLiSibling = nextLiSibling.nextSibling;
    }
    if (nextLiSibling) {
      as = nextLiSibling.querySelectorAll('a');
      if (as.length > 0)
        nextElement = as[0];
    }

    if (nextElement) {
      next.classList.remove('disabled');
      next.setAttribute('href', nextElement.getAttribute('href'));
      addDynamicLoadEvent(next);
    } else
      next.classList.add('disabled');
  }

  if (previous) {
    var previousElement = null;

    var previousLiSibling = selected.previousSibling;
    while (previousLiSibling) {
      if (previousLiSibling.tagName && previousLiSibling.tagName.toLowerCase() === 'li')
        break;
      previousLiSibling = previousLiSibling.previousSibling;
    }
    if (previousLiSibling) {
      as = previousLiSibling.querySelectorAll('a');
      if (as.length > 0)
        previousElement = as[0];
    }

    if (previousElement) {
      previous.classList.remove('disabled');
      previous.setAttribute('href', previousElement.getAttribute('href'));
      addDynamicLoadEvent(previous);
    } else
      previous.classList.add('disabled');
  }

  if (up) {
    var upElement = null;
    var parentLi = null;
    if (selected.parentNode.parentNode.tagName.toLowerCase() === 'li')
      parentLi = selected.parentNode.parentNode;
    if (parentLi) {
      as = parentLi.querySelectorAll('a');
      if (as.length > 0)
        upElement = as[0];
    }

    if (upElement) {
      up.classList.remove('disabled');
      up.setAttribute('href', upElement.getAttribute('href'));
      addDynamicLoadEvent(up);
    } else {
      up.setAttribute('href', forgeUrl(localSetup.book, 'index'));
      addDynamicLoadEvent(up);
      up.classList.remove('disabled');
    }
  }
}

function populateMenu(menu) {
  // make in order that the <li> tags above the <a> are also clickable
  var lis = menu.querySelectorAll('li');
  for (var i = 0; i < lis.length; i++) {
    var li = lis[i];
    li.addEventListener('click',
      function(event) {
        var as = event.target.querySelectorAll('a');
        if (as.length > 0)
          aClick(as[0]);
      }
    );
  }

  var menuDiv = document.querySelector('#menu');
  menuDiv.appendChild(menu);

  menu.setAttribute('id', 'accordion');
  $('#accordion > li > a').click(function() {
    showAccodionItem(this);
  });
}

function showAccodionItem(item) {
  if (!$(item).hasClass('active')) {
    $('#accordion li ul').slideUp();
    $(item).next().slideToggle();
    $('#accordion li a').removeClass('active');
    $(item).addClass('active');
  }
}

function getMDFile() {
  var target = computeTargetPath() + localSetup.page + '.md';
  console.log('Get MD file: ' + target);
  $.ajax({
    type: 'GET',
    url: target,
    dataType: 'text',
    success: populateViewDiv,
    error: function(XMLHttpRequest, textStatus, errorThrown) {
      console.log('Status: ' + textStatus);
      console.log('Error: ' + errorThrown);
      var mainPage = 'index';
      // get the main page instead
      if (localSetup.page !== mainPage) {
        localSetup.page = mainPage;
        getMDFile();
      }
    }
  });
}

function getMenuFile() {
  var target = computeTargetPath() + 'menu.md';
  console.log('Get menu file: ' + target);
  $.ajax({
    type: 'GET',
    url: target,
    dataType: 'text',
    success: receiveMenuContent,
    error: function(XMLHttpRequest, textStatus, errorThrown) {
      console.log('Status: ' + textStatus);
      console.log('Error: ' + errorThrown);
    }
  });
}

function extractAnchor(url) {
  var match = /#([\w-]+)/.exec(url);
  if (match && match.length === 2)
    return match[1];
  return '';
}

// width: in pixels
function setHandleWidth(width) {
  handle.left.css('width', width + 'px');
  handle.menu.css('width', width + 'px');
  handle.handle.css('left', width + 'px');
  handle.center.css('left', width + 'px');
  handle.center.css('width', 'calc(100% - ' + width + 'px)');
}

function initializeHandle() {
  // inspired from: http://stackoverflow.com/questions/17855401/how-do-i-make-a-div-width-draggable
  handle = {}; // structure where all the handle info is stored

  handle.left = $('#left');
  handle.menu = $('#menu');
  handle.center = $('#center');
  handle.handle = $('#handle');
  handle.container = $('#opendr-doc');

  // dimension bounds of the handle in pixels
  handle.min = 0;
  handle.minThreshold = 90; // under this threshold, the handle is totally hidden
  handle.initialWidth = Math.max(handle.minThreshold, handle.left.width());
  handle.max = Math.max(250, handle.initialWidth);

  handle.isResizing = false;
  handle.lastDownX = 0;

  if (isOpenDRUrl) {
    handle.left.addClass('opendr');
    handle.handle.addClass('opendr');
    handle.center.addClass('opendr');
  } else {
    handle.left.addClass('default');
    handle.handle.addClass('default');
    handle.center.addClass('default');
  }

  setHandleWidth(handle.initialWidth);

  handle.handle.on('mousedown touchstart', function(e) {
    if (e.type === 'touchstart')
      e = e.originalEvent.touches[0];
    handle.isResizing = true;
    handle.lastDownX = e.clientX;
    handle.container.css('user-select', 'none');
  }).on('dblclick', function(e) {
    if (handle.left.css('width').startsWith('0'))
      setHandleWidth(handle.initialWidth);
    else
      setHandleWidth(0);
  });

  $(document).on('mousemove touchmove', function(e) {
    if (e.type === 'touchmove')
      e = e.originalEvent.touches[0];
    if (!handle.isResizing)
      return;
    var mousePosition = e.clientX - handle.container.offset().left; // in pixels
    if (mousePosition < handle.minThreshold / 2) {
      setHandleWidth(0);
      return;
    } else if (mousePosition < handle.minThreshold)
      return;
    if (mousePosition < handle.min || mousePosition > handle.max)
      return;
    setHandleWidth(mousePosition);
  }).on('mouseup touchend', function(e) {
    handle.isResizing = false;
    handle.container.css('user-select', 'auto');
  });
}

window.onscroll = function() {
  if (!isOpenDRUrl)
    return;
  updateMenuScrollbar();
};

document.addEventListener('DOMContentLoaded', function() {
  //window.mermaidAPI.initialize({startOnLoad: false});
  initializeHandle();

  if (!isOpenDRUrl) {
    if (!localSetup.url)
      localSetup.url = 'http://localhost:8000/'; //getGETQueryValue('url', 'https://raw.githubusercontent.com/opendr/opendr/');
    if (!localSetup.book)
      localSetup.book = getGETQueryValue('book', 'reference');
    if (!localSetup.page)
      localSetup.page = getGETQueryValue('page', 'index');
    if (!localSetup.anchor)
      localSetup.anchor = window.location.hash.substring(1);
    if (!localSetup.branch)
      localSetup.branch = ''; // getGETQueryValue('branch', 'master');
    if (!localSetup.tabs)
      localSetup.tabs = getGETQueriesMatchingRegularExpression('^tab-\\w+$', 'g');
    // backward compatibility <= R2019b revision 1
    if (!localSetup.tabs['tab-language']) {
      if (localSetup.tab) {
        localSetup.tabs['tab-language'] = localSetup.tab;
        delete localSetup.tab;
      } else
        localSetup.tabs['tab-language'] = getGETQueryValue('tab', '').toLowerCase();
    }
  }

  setupModalWindow();
  applyToTitleDiv();
  getMDFile();
  getMenuFile();
});
