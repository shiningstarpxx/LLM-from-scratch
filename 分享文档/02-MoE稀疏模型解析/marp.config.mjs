import { Marp } from '@marp-team/marp-core'
import markdownItMermaid from 'markdown-it-mermaid'

export default {
  engine: (opts) => {
    const marp = new Marp(opts)
    
    // 👇 兼容性写法：如果有 .default 就用，没有就用本身
    const mermaidPlugin = markdownItMermaid.default || markdownItMermaid
    marp.use(mermaidPlugin)
    
    return marp
  },
}

