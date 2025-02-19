<template>
    <div class="manual-page">
        <!-- Sidebar: Markdown dosyalarının listesi -->
        <div class="manual-sidebar">
            <h2>Dosyalar</h2>
            <ul>
                <li
                    v-for="file in files"
                    :key="file"
                    :class="{ active: activeFile === file }"
                    @click="loadMarkdown(file)"
                >
                    {{ file }}
                </li>
            </ul>
        </div>

        <!-- İçerik: Render edilmiş markdown -->
        <div class="manual-content" v-html="activeFileContent"></div>
    </div>
</template>

<script>
import { marked } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/github.css";

// Marked için highlight.js entegrasyonu:
// Eğer marked v4+ kullanıyorsanız, marked.use() metodunu kullanabilirsiniz.
marked.use({
    highlight: function (code, lang) {
        const language = hljs.getLanguage(lang) ? lang : "plaintext";
        return hljs.highlight(code, { language }).value;
    }
});

export default {
    name: "ManualPage",
    data() {
        return {
            files: [ "intro.md", "usage.md", "advanced.md" ],
            activeFile: "",
            activeFileContent: ""
        };
    },
    mounted() {
        if (this.files.length > 0) {
            this.loadMarkdown(this.files[0]);
        }
    },
    methods: {
        loadMarkdown(file) {
            this.activeFile = file;
            fetch(`/manuals/${file}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Dosya yüklenemedi: ${file}`);
                    }
                    return response.text();
                })
                .then(markdown => {
                    // Marked parse fonksiyonunu kullanarak markdown'u HTML'e dönüştürüyoruz.
                    this.activeFileContent = marked.parse(markdown);
                    // HTML render edildikten sonra, varsa kod bloklarını highlight.js ile yeniden işleyelim.
                    this.$nextTick(() => {
                        document.querySelectorAll("pre code").forEach((block) => {
                            hljs.highlightElement(block);
                        });
                    });
                })
                .catch(error => {
                    console.error(error);
                    this.activeFileContent = "<p>Dosya yüklenirken hata oluştu.</p>";
                });
        }
    }
};
</script>

  <style scoped>
  .manual-page {
    display: flex;
    padding: 20px;
  }

  /* Sidebar stil ayarları */
  .manual-sidebar {
    width: 200px;
    margin-right: 20px;
    border-right: 1px solid #ddd;
    padding-right: 10px;
  }

  .manual-sidebar ul {
    list-style: none;
    padding: 0;
  }

  .manual-sidebar li {
    padding: 5px 0;
    cursor: pointer;
  }

  .manual-sidebar li.active {
    font-weight: bold;
  }

  /* Markdown içeriği stil ayarları */
  .manual-content {
    flex: 1;
    font-size: 16px;
    line-height: 1.6;
  }
  </style>
