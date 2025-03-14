<template>
    <div class="manual-page">
      <!-- Sidebar: List of Markdown files -->
      <div class="manual-sidebar">
        <h2>Files</h2>
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
      <!-- Content: Rendered markdown -->
      <div class="manual-content" v-html="activeFileContent"></div>
    </div>
  </template>
  
  <script>
  import { marked } from "marked";
  import hljs from "highlight.js";
  import "highlight.js/styles/github.css";
  
  // Integration of highlight.js for Marked:
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
        files: [],
        activeFile: "",
        activeFileContent: "",
        pollingInterval: null
      };
    },
    mounted() {
      // Initial fetch
      this.fetchMarkdownFilesList();
      
      // Set up polling every 30 seconds
      this.pollingInterval = setInterval(() => {
        this.fetchMarkdownFilesList(true);
      }, 30000);
    },
    beforeUnmount() {
      // Clean up the interval when component is destroyed
      if (this.pollingInterval) {
        clearInterval(this.pollingInterval);
      }
    },
    methods: {
      fetchMarkdownFilesList(silent = false) {
        // Fetch the list of available markdown files
        fetch('/manuals/index.json')
          .then(response => {
            if (!response.ok) {
              throw new Error('Could not load file index');
            }
            return response.json();
          })
          .then(data => {
            // Compare new files with existing ones
            const newFiles = data.files;
            const hasChanges = JSON.stringify(newFiles) !== JSON.stringify(this.files);
            
            if (hasChanges) {
              this.files = newFiles;
              if (!this.activeFile && this.files.length > 0) {
                this.loadMarkdown(this.files[0]);
              }
              if (!silent) {
                console.log('Manual files updated:', this.files);
              }
            }
          })
          .catch(error => {
            if (!silent) {
              console.error('Error loading file index:', error);
              // Fallback to default files
              this.files = ["intro.md", "usage.md", "advanced.md"];
              if (!this.activeFile && this.files.length > 0) {
                this.loadMarkdown(this.files[0]);
              }
            }
          });
      },
      loadMarkdown(file) {
        this.activeFile = file;
        fetch(`/manuals/${file}`)
          .then(response => {
            if (!response.ok) {
              throw new Error(`File could not be loaded: ${file}`);
            }
            return response.text();
          })
          .then(markdown => {
            // Convert markdown to HTML using Marked's parse function.
            this.activeFileContent = marked.parse(markdown);
            // After the HTML is rendered, reprocess any code blocks with highlight.js.
            this.$nextTick(() => {
              document.querySelectorAll("pre code").forEach((block) => {
                hljs.highlightElement(block);
              });
            });
          })
          .catch(error => {
            console.error(error);
            this.activeFileContent = "<p>An error occurred while loading the file.</p>";
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
  /* Sidebar styles */
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
  /* Markdown content styles */
  .manual-content {
    flex: 1;
    font-size: 16px;
    line-height: 1.6;
  }
  </style>