import tkinter as tk
from tkinter import scrolledtext, messagebox, font, filedialog
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations
import numpy as np
import os 

class AppPCA(tk.Tk):

    def __init__(self):
        super().__init__()

        # --- Configuración de la Ventana ---
        self.title("Análisis de Componentes Principales (PCA)")
        self.attributes('-fullscreen', True)

        # --- Colores y Fuentes ---
        self.BG_COLOR = "#1a1a2e"
        self.FG_COLOR = "#FFFFFF"
        self.TEXT_BG = "#2e2e5c"
        self.BTN_COLOR = "#9a7fdd"
        self.BTN_ACTIVE = "#7a5fbd"
        self.BTN_EXIT_BG = "#dc3545"
        self.SUCCESS_COLOR = "#4CAF50"
        self.WARN_COLOR = "#F44336"

        self.TITLE_FONT = ("Helvetica", 22, "bold")
        self.DESC_FONT = ("Helvetica", 12)
        self.BUTTON_FONT = ("Helvetica", 11, "bold")
        self.RESULT_FONT = ("Consolas", 11)
        self.FINAL_FONT = ("Consolas", 13, "bold")
        self.SIDEBAR_FONT = ("Helvetica", 10) # Fuente para checkboxes

        self.configure(bg=self.BG_COLOR)

        # --- Almacenamiento de datos ---
        self.cov_matrix_df = None
        self.pca_cov_matrix_df = None
        self.data_transformed = None
        self.n_components_pca = 0
        self.spin_n_components = None
        
        # --- NUEVAS variables para el flujo ---
        self.data_raw = None           # Almacenará el DataFrame cargado
        self.all_column_names = []   # Lista de todas las columnas del CSV
        self.column_vars = []        # Lista para las tk.BooleanVar de los checkboxes
        self.btn_analyze = None        # Referencia al nuevo botón "Analizar"
        self.loaded_filepath = None # <-- 2. AÑADIR PARA GUARDAR RUTA
        self.selected_variable_count = 0 # <-- 2. AÑADIR PARA GUARDAR CONTEO
        
        # --- NUEVAS variables para layout ---
        self.sidebar_frame = None
        self.canvas_sidebar = None
        self.checkbox_container = None

        # --- Bindings ---
        self.bind("<Escape>", self.cerrar_app)

        # --- Inicializar la Interfaz ---
        self.crear_widgets()

    def cerrar_app(self, event=None):
        """Cierra la aplicación."""
        self.destroy()

    def _validate_spinbox_input(self, proposed_value):
        """Valida que la entrada del Spinbox sea un número entre 1 y n_components_pca (o 10)."""
        if proposed_value == "":
            return True
        try:
            value = int(proposed_value)
            max_comps = self.n_components_pca if self.n_components_pca > 0 else 10
            if 1 <= value <= max_comps:
                return True
            else:
                self.bell()
                return False
        except ValueError:
            self.bell()
            return False

    # --- NUEVA FUNCIÓN (Helper para scroll) ---
    def on_frame_configure(self, event=None):
        """Actualiza el scrollregion del canvas para abarcar el frame de checkboxes."""
        if self.canvas_sidebar:
            self.canvas_sidebar.configure(scrollregion=self.canvas_sidebar.bbox("all"))

    # --- FUNCIÓN MODIFICADA: Layout y botones ---
    def crear_widgets(self):
        """Crea y posiciona todos los widgets en la ventana."""

        # --- Crear frames de layout ---
        top_frame = tk.Frame(self, bg=self.BG_COLOR)
        top_frame.pack(side="top", fill="x", pady=(30, 10))

        bottom_frame = tk.Frame(self, bg=self.BG_COLOR)
        bottom_frame.pack(side="bottom", fill="x", pady=(10, 20))

        # --- NUEVO: Main frame para dividir 1/4 y 3/4 ---
        main_frame = tk.Frame(self, bg=self.BG_COLOR)
        main_frame.pack(side="top", fill="both", expand=True, padx=50, pady=20)

        # --- NUEVO: Panel lateral (Sidebar) ---
        # Usamos winfo_screenwidth para calcular 1/4 de la pantalla
        try:
            screen_width = self.winfo_screenwidth()
        except tk.TclError: # Fallback por si la ventana no está lista
            screen_width = 1920 
            
        sidebar_width = screen_width // 4

        self.sidebar_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=sidebar_width)
        self.sidebar_frame.pack(side="left", fill="y", padx=(0, 20))
        # Evita que el frame se encoja al tamaño de sus hijos
        self.sidebar_frame.pack_propagate(False)

        # --- NUEVO: Panel de resultados (ocupa el resto) ---
        results_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        results_frame.pack(side="left", fill="both", expand=True)

        # --- Widgets del Frame Superior ---
        lbl_title = tk.Label(top_frame,
                             text="Analizador de componentes principales",
                             font=self.TITLE_FONT,
                             bg=self.BG_COLOR,
                             fg=self.FG_COLOR)
        lbl_title.pack(pady=(10, 10))

        # --- Frame para el selector de componentes ---
        n_components_frame = tk.Frame(top_frame, bg=self.BG_COLOR)
        n_components_frame.pack(pady=5)

        lbl_n_components = tk.Label(n_components_frame,
                                       text="Número de componentes a retener (para cálculo de pérdida):",
                                       font=self.DESC_FONT,
                                       bg=self.BG_COLOR,
                                       fg=self.FG_COLOR)
        lbl_n_components.pack(side="left", padx=(0, 10))

        self.n_components_var = tk.StringVar(value="1")
        vcmd = (self.register(self._validate_spinbox_input), '%P')

        self.spin_n_components = tk.Spinbox(n_components_frame,
                                            from_=1,
                                            to=10,
                                            textvariable=self.n_components_var,
                                            width=4,
                                            font=self.DESC_FONT,
                                            bg=self.TEXT_BG,
                                            fg=self.FG_COLOR,
                                            relief=tk.FLAT,
                                            bd=0,
                                            buttonbackground=self.BTN_COLOR,
                                            readonlybackground=self.TEXT_BG,
                                            validate='key',
                                            validatecommand=vcmd)
        self.spin_n_components.pack(side="left")

        # --- Frame para botones horizontales ---
        button_bar_frame = tk.Frame(top_frame, bg=self.BG_COLOR)
        button_bar_frame.pack(pady=(10, 5))
        
        # ... (botones de matriz) ...
        self.btn_ver_matriz = tk.Button(button_bar_frame,
                                 text="Ver matriz (Original)",
                                 font=self.BUTTON_FONT, bg=self.BTN_COLOR, fg=self.FG_COLOR,
                                 activebackground=self.BTN_ACTIVE, activeforeground=self.FG_COLOR,
                                 command=self.mostrar_ventana_matriz_original,
                                 relief=tk.FLAT, padx=15, pady=10, bd=0, state=tk.DISABLED)
        self.btn_ver_matriz.pack(side="left", padx=10)

        self.btn_ver_pca_matriz = tk.Button(button_bar_frame,
                                 text="Ver matriz (PCA)",
                                 font=self.BUTTON_FONT, bg=self.BTN_COLOR, fg=self.FG_COLOR,
                                 activebackground=self.BTN_ACTIVE, activeforeground=self.FG_COLOR,
                                 command=self.mostrar_ventana_matriz_pca,
                                 relief=tk.FLAT, padx=15, pady=10, bd=0, state=tk.DISABLED)
        self.btn_ver_pca_matriz.pack(side="left", padx=10)

        # --- BOTÓN MODIFICADO: Solo carga ---
        btn_load = tk.Button(button_bar_frame,
                                 text="Cargar Archivo CSV",
                                 font=self.BUTTON_FONT,
                                 bg=self.BTN_COLOR,
                                 fg=self.FG_COLOR,
                                 activebackground=self.BTN_ACTIVE,
                                 activeforeground=self.FG_COLOR,
                                 command=self.cargar_archivo, # <-- CAMBIO DE COMANDO
                                 relief=tk.FLAT,
                                 padx=15,
                                 pady=10,
                                 bd=0)
        btn_load.pack(side="left", padx=10)

        # --- NUEVO BOTÓN: Analizar ---
        self.btn_analyze = tk.Button(button_bar_frame,
                                 text="Analizar variables",
                                 font=self.BUTTON_FONT,
                                 bg=self.SUCCESS_COLOR, # Color diferente para destacar
                                 fg=self.FG_COLOR,
                                 activebackground="#388E3C",
                                 activeforeground=self.FG_COLOR,
                                 command=self.iniciar_analisis_con_seleccion, # <-- NUEVO COMANDO
                                 relief=tk.FLAT,
                                 padx=15,
                                 pady=10,
                                 bd=0,
                                 state=tk.DISABLED) # Inicia deshabilitado
        self.btn_analyze.pack(side="left", padx=10)

        # --- NUEVO BOTÓN: Buscador de Combinaciones ---
        self.btn_combinations = tk.Button(button_bar_frame,
                                         text="🔍 Buscar Mejores Combinaciones",
                                         font=self.BUTTON_FONT,
                                         bg="#FF9800", # Naranja para diferenciar
                                         fg=self.FG_COLOR,
                                         activebackground="#F57C00",
                                         activeforeground=self.FG_COLOR,
                                         command=self.abrir_ventana_combinaciones,
                                         relief=tk.FLAT,
                                         padx=15,
                                         pady=10,
                                         bd=0,
                                         state=tk.DISABLED) # Inicia deshabilitado
        self.btn_combinations.pack(side="left", padx=10)


        # --- Botón para Guardar Datos ---
        self.btn_guardar_pca = tk.Button(button_bar_frame,
                                 text="Guardar datos PCA (Excel)",
                                 font=self.BUTTON_FONT,
                                 bg="#5bc0de", # Color cian
                                 fg=self.FG_COLOR,
                                 activebackground="#31b0d5",
                                 activeforeground=self.FG_COLOR,
                                 command=self.guardar_datos_pca,
                                 relief=tk.FLAT, padx=15, pady=10, bd=0, state=tk.DISABLED)
        self.btn_guardar_pca.pack(side="left", padx=10)

        # --- Widgets del Panel Lateral (Sidebar) ---
        lbl_sidebar_title = tk.Label(self.sidebar_frame,
                                     text="Variables del archivo",
                                     font=self.TITLE_FONT,
                                     bg=self.BG_COLOR,
                                     fg=self.BTN_COLOR)
        lbl_sidebar_title.pack(side="top", fill="x", pady=(0, 10))
        
        lbl_sidebar_desc = tk.Label(self.sidebar_frame,
                                     text="Seleccione las variables a incluir:",
                                     font=self.DESC_FONT,
                                     bg=self.BG_COLOR,
                                     fg=self.FG_COLOR)
        lbl_sidebar_desc.pack(side="top", fill="x", pady=(0, 10))

        # --- NUEVO: Contenedor scrollable para checkboxes ---
        container_frame = tk.Frame(self.sidebar_frame, bg=self.TEXT_BG)
        container_frame.pack(fill="both", expand=True, padx=5, pady=5)
        container_frame.config(highlightbackground=self.BTN_COLOR, highlightthickness=1)

        self.canvas_sidebar = tk.Canvas(container_frame, bg=self.TEXT_BG, highlightthickness=0)
        scrollbar_sidebar = tk.Scrollbar(container_frame, orient="vertical", command=self.canvas_sidebar.yview)

        self.canvas_sidebar.configure(yscrollcommand=scrollbar_sidebar.set)

        scrollbar_sidebar.pack(side="right", fill="y")
        self.canvas_sidebar.pack(side="left", fill="both", expand=True)
        
         # Este es el frame que contendrá los checkboxes, DENTRO del canvas
        self.checkbox_container = tk.Frame(self.canvas_sidebar, bg=self.TEXT_BG)
        
        self.canvas_sidebar.create_window((0, 0), window=self.checkbox_container, anchor="nw")
        
        # Vincular el cambio de tamaño del frame al canvas
        self.checkbox_container.bind("<Configure>", self.on_frame_configure)


        # --- Widgets del Frame de Resultados ---
        self.txt_results = scrolledtext.ScrolledText(results_frame,
                                                       wrap=tk.WORD,
                                                       font=self.RESULT_FONT,
                                                       bg=self.TEXT_BG,
                                                       fg=self.FG_COLOR,
                                                       relief=tk.FLAT,
                                                       borderwidth=0,
                                                       insertbackground=self.FG_COLOR)
        self.txt_results.pack(expand=True, fill="both")


        # --- Widgets del Frame Inferior ---
        btn_exit = tk.Button(bottom_frame,
                                 text="Salir",
                                 font=self.BUTTON_FONT, bg=self.BTN_EXIT_BG, fg=self.FG_COLOR,
                                 activebackground="#b72a38", activeforeground=self.FG_COLOR,
                                 command=self.cerrar_app,
                                 relief=tk.FLAT, padx=15, pady=10, bd=0)
        btn_exit.pack()

        # --- Etiquetas de estilo para el texto ---
        self.txt_results.tag_config('title', font=(self.RESULT_FONT[0], 12, "bold"), foreground=self.BTN_COLOR, spacing1=5, spacing3=5)
        self.txt_results.tag_config('success', font=self.FINAL_FONT, foreground=self.SUCCESS_COLOR)
        self.txt_results.tag_config('warning', font=self.FINAL_FONT, foreground=self.WARN_COLOR)
        self.txt_results.tag_config('info', font=(self.RESULT_FONT[0], 10), foreground="#D1C4E9", lmargin1=10, lmargin2=10)

        self.txt_results.insert(tk.END, "Presione 'Cargar Archivo CSV' para seleccionar un archivo...")
        self.txt_results.configure(state='disabled')


    # --- NUEVA FUNCIÓN ---
    def cargar_archivo(self):
        """
        Abre un diálogo para que el usuario seleccione un archivo CSV.
        Si se selecciona, carga los datos y puebla la lista de variables.
        """
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo CSV para analizar",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )

        if not filepath:
            messagebox.showinfo("Cancelado", "No se seleccionó ningún archivo.")
            return

        self.loaded_filepath = filepath # <-- 3. GUARDAR LA RUTA DEL ARCHIVO

        try:
            # Cargar y guardar el DataFrame
            self.data_raw = pd.read_csv(filepath)
            self.all_column_names = self.data_raw.columns.tolist()

            # Poblar la sidebar con las columnas
            self.poblar_sidebar()

            # Habilitar el botón de análisis
            if self.btn_analyze:
                self.btn_analyze.config(state=tk.NORMAL)
            
            # Informar al usuario en el panel de resultados
            self.txt_results.configure(state='normal')
            self.txt_results.delete(1.0, tk.END)
            self.txt_results.insert(tk.END, f"Archivo cargado: {filepath}\n\n")
            self.txt_results.insert(tk.END, f"Se encontraron {len(self.all_column_names)} columnas.\n")
            self.txt_results.insert(tk.END, "Por favor, seleccione las variables a analizar en el panel izquierdo y presione 'Analizar Variables'.\n")
            self.txt_results.configure(state='disabled')

            # Deshabilitar botones de resultados previos
            self.btn_ver_matriz.config(state=tk.DISABLED)
            self.btn_ver_pca_matriz.config(state=tk.DISABLED)
            self.btn_guardar_pca.config(state=tk.DISABLED)
            self.btn_combinations.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error al leer CSV", f"No se pudo leer el archivo CSV:\n{e}")
            self.txt_results.configure(state='normal')
            self.txt_results.delete(1.0, tk.END)
            self.txt_results.insert(tk.END, f"ERROR al leer {filepath}: {e}\n")
            self.txt_results.configure(state='disabled')
            if self.btn_analyze:
                self.btn_analyze.config(state=tk.DISABLED)


    # --- NUEVA FUNCIÓN ---
    def poblar_sidebar(self):
        """Limpia y puebla el sidebar con checkboxes para cada columna."""
        # Limpiar widgets anteriores
        for widget in self.checkbox_container.winfo_children():
            widget.destroy()
        
        self.column_vars = []
        
        # Estilo para los checkboxes
        style = {
            'bg': self.TEXT_BG,
            'fg': self.FG_COLOR,
            'selectcolor': self.BG_COLOR, # Color de fondo de la marca
            'activebackground': self.BG_COLOR,
            'activeforeground': self.FG_COLOR,
            'font': self.SIDEBAR_FONT,
            'relief': tk.FLAT,
            'bd': 0,
            'highlightthickness': 0
        }

        for i, col_name in enumerate(self.all_column_names):
            var = tk.BooleanVar()
            
            # Heurística: Pre-seleccionar columnas C a L (índices 2 a 11) por defecto
            # Esto mantiene el comportamiento original si se carga el mismo tipo de archivo
            if 2 <= i <= 11:
                var.set(True)
            else:
                var.set(False)
            
            cb = tk.Checkbutton(self.checkbox_container, 
                                 text=col_name, 
                                 variable=var, 
                                 anchor="w",
                                 **style)
            cb.pack(side="top", fill="x", padx=10, pady=2)
            self.column_vars.append(var)
        
        # Resetear la vista del scroll y actualizar
        self.canvas_sidebar.yview_moveto(0)
        self.on_frame_configure()


    # --- FUNCIÓN MODIFICADA: Ahora solo llama al análisis ---
    def iniciar_analisis_con_seleccion(self):
        """
        Inicia el análisis de PCA usando las variables seleccionadas 
        en el sidebar.
        """
        if self.data_raw is None:
            messagebox.showwarning("Sin Datos", "Por favor, cargue un archivo primero.")
            return

        # Obtener las variables seleccionadas de los checkboxes
        selected_column_names = []
        for i, var in enumerate(self.column_vars):
            if var.get():
                selected_column_names.append(self.all_column_names[i])

        if len(selected_column_names) == 0:
            messagebox.showwarning("Sin Selección", "Por favor, seleccione al menos una variable para analizar.")
            return
        
        self.selected_variable_count = len(selected_column_names) # <-- 4. GUARDAR EL CONTEO

        # Pasa las columnas seleccionadas a la función de análisis
        self.ejecutar_analisis(selected_column_names)


    # --- FUNCIÓN MODIFICADA: Recibe las columnas a analizar ---
    def ejecutar_analisis(self, selected_columns):
        """
        Función principal que se ejecuta al presionar "Analizar Variables".
        Usa el DataFrame cargado (self.data_raw) y las columnas seleccionadas.
        
        Args:
            selected_columns (list): La lista de nombres de columnas a analizar.
        """
        try:
            # Deshabilitar botones al iniciar análisis
            self.btn_ver_matriz.config(state=tk.DISABLED)
            self.btn_ver_pca_matriz.config(state=tk.DISABLED)
            self.btn_guardar_pca.config(state=tk.DISABLED)
            self.cov_matrix_df = None
            self.pca_cov_matrix_df = None
            self.data_transformed = None
            self.n_components_pca = 0

            # Habilitar texto para escritura
            self.txt_results.configure(state='normal')
            self.txt_results.delete(1.0, tk.END)
            
            self.txt_results.insert(tk.END, f"Iniciando análisis con {len(selected_columns)} variables seleccionadas...\n\n")
            self.update_idletasks()

            # --- 1. Limpieza de Datos (YA CARGADOS) ---
            
            # Quitar columnas 'Unnamed' si existen (buena práctica)
            data_raw_cleaned = self.data_raw.loc[:, ~self.data_raw.columns.str.contains('^Unnamed')]
            
            # Verificar si las columnas seleccionadas existen
            try:
                # Usar .copy() para evitar SettingWithCopyWarning
                data_numeric = data_raw_cleaned[selected_columns].copy()
            except KeyError as e:
                messagebox.showerror("Error de Columnas", f"Una de las columnas seleccionadas ({e}) no se pudo encontrar. Recargue el archivo.")
                self.txt_results.insert(tk.END, f"ERROR: Columna no encontrada {e}.\n")
                return
            except Exception as e:
                messagebox.showerror("Error de Selección", f"Ocurrió un error al seleccionar las columnas: {e}")
                self.txt_results.insert(tk.END, f"ERROR: {e}.\n")
                return
                
            column_names = data_numeric.columns.tolist()

            # Forzar conversión a numérico y manejar errores
            for col in data_numeric.columns:
                data_numeric[col] = pd.to_numeric(data_numeric[col], errors='coerce')

            original_rows = len(data_numeric)
            # data_numeric ya fue 'droppeada' de NAs en la carga, pero volvemos a hacerlo por si acaso
            rows_before_drop = len(data_numeric)
            data_numeric = data_numeric.dropna()
            cleaned_rows = len(data_numeric)
            rows_dropped = rows_before_drop - cleaned_rows

            self.txt_results.insert(tk.END, f"Se cargaron {original_rows} filas (para estas variables).\n")
            if rows_dropped > 0:
                self.txt_results.insert(tk.END, f"Se eliminaron {rows_dropped} filas con datos faltantes o no numéricos.\n", 'warning')
            if cleaned_rows == 0:
                messagebox.showerror("Error de Datos", "No quedaron filas válidas después de limpiar los datos. Revise las variables seleccionadas.")
                self.txt_results.insert(tk.END, f"ERROR: No hay datos válidos para analizar.\n")
                return

            self.txt_results.insert(tk.END, f"Analizando {cleaned_rows} filas y {len(column_names)} variables:\n")
            self.txt_results.insert(tk.END, ", ".join(column_names) + "\n\n")

            # --- 2. Matriz de Varianza-Covarianza (Datos Originales) ---
            self.txt_results.insert(tk.END, "--- 1. Matriz de varianza-covarianza (Datos originales) ---\n", 'title')
            cov_matrix = data_numeric.cov()
            self.cov_matrix_df = cov_matrix
            self.txt_results.insert(tk.END, self.cov_matrix_df.to_string(float_format="%.4f") + "\n\n")

            # --- 3. Estandarización ---
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_numeric)

            # --- 4. Análisis de Componentes Principales (PCA) ---
            self.txt_results.insert(tk.END, "--- 2. Análisis de Componentes Principales (sobre datos estandarizados) ---\n", 'title')
            
            # Asegurarse de que n_components no sea mayor que las variables
            n_features = data_scaled.shape[1]
            if n_features == 0:
                messagebox.showerror("Error", "No hay variables (features) para analizar.")
                return

            # PCA() por defecto usa min(n_muestras, n_variables)
            pca = PCA() 
            self.data_transformed = pca.fit_transform(data_scaled)
            self.n_components_pca = pca.n_components_
            
            # --- Actualizar 'to' del Spinbox ---
            if self.spin_n_components:
                # El máximo de componentes es el número de variables que entran
                max_comps_posibles = min(cleaned_rows, len(column_names))
                self.spin_n_components.config(to=max_comps_posibles)
                try:
                    current_val = int(self.n_components_var.get())
                    if current_val > max_comps_posibles:
                        self.n_components_var.set(str(max_comps_posibles))
                except ValueError:
                    self.n_components_var.set("1")

            explained_variance = pca.explained_variance_ratio_

            self.txt_results.insert(tk.END, "Varianza explicada por cada componente:\n")
            for i, var in enumerate(explained_variance):
                self.txt_results.insert(tk.END, f"  Componente principal {i+1}: {var*100:6.2f}%\n")

            # --- 5. Matriz de Covarianza de PCA ---
            self.txt_results.insert(tk.END, "\n--- 3. Matriz de covarianza (Nuevas variables PCA) ---\n", 'title')
            self.txt_results.insert(tk.END,
                                 "Esta matriz muestra que las nuevas variables (Componentes Principales) no están correlacionadas entre sí (valores fuera de la diagonal son ~0).\nLa diagonal muestra la varianza de cada componente.\n\n",
                                 'info')

            pca_cov_matrix = np.cov(self.data_transformed, rowvar=False)
            pc_names = [f"PC{i+1}" for i in range(self.n_components_pca)]
            self.pca_cov_matrix_df = pd.DataFrame(pca_cov_matrix, columns=pc_names, index=pc_names)

            self.txt_results.insert(tk.END, self.pca_cov_matrix_df.to_string(float_format="%.4f") + "\n\n")

            # --- 6. Resultado de Reducción ---
            try:
                n_comps_display = int(self.n_components_var.get())
                max_comps_spinbox = int(self.spin_n_components.cget('to'))
                if not (1 <= n_comps_display <= max_comps_spinbox):
                    messagebox.showwarning("Valor inválido", f"El número de componentes debe estar entre 1 y {max_comps_spinbox}. Usando 1.")
                    n_comps_display = 1
                    self.n_components_var.set("1")
            except ValueError:
                messagebox.showwarning("Valor Inválido", "El número de componentes debe ser un número entero. Usando 1.")
                n_comps_display = 1
                self.n_components_var.set("1")

            self.txt_results.insert(tk.END, f"\n--- 4. Resultado de reducción a {n_comps_display} Componente(s) (para cálculo) ---\n", 'title')

            info_kept = np.sum(explained_variance[:n_comps_display]) * 100
            info_lost = 100.0 - info_kept

            s_comps = "s" if n_comps_display > 1 else ""
            s_pc = f"PC1 a PC{n_comps_display}" if n_comps_display > 1 else "PC1"

            self.txt_results.insert(tk.END, f"\nPorcentaje de información (varianza) retenida por {s_pc}: \n", 'success')
            self.txt_results.insert(tk.END, f"{info_kept:.2f}%\n\n", 'success')

            self.txt_results.insert(tk.END, f"Porcentaje de información (varianza) PERDIDA al reducir a {n_comps_display} componente{s_comps}: \n", 'warning')
            self.txt_results.insert(tk.END, f"{info_lost:.2f}%\n", 'warning')

            # Habilitar TODOS los botones de resultados
            self.btn_ver_matriz.config(state=tk.NORMAL)
            self.btn_ver_pca_matriz.config(state=tk.NORMAL)
            self.btn_guardar_pca.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error inesperado durante el análisis: {e}")
            self.txt_results.insert(tk.END, f"ERROR INESPERADO: {e}\n")

        finally:
            self.txt_results.configure(state='disabled')

    def guardar_datos_pca(self):
        """
        Guarda TODOS los datos transformados por PCA en un nuevo archivo XLSX (Excel).
        AHORA TAMBIÉN GUARDA LA SUMA (Y USA NOMBRE AUTOMÁTICO).
        """
        if self.data_transformed is None or self.n_components_pca == 0:
            messagebox.showwarning("Sin Datos", "No hay datos transformados para guardar. Ejecute el análisis primero.")
            return

        n_comps_to_save = self.n_components_pca

        # --- 5. MODIFICACIÓN PARA NOMBRE AUTOMÁTICO ---
        if self.loaded_filepath:
            # Obtener el nombre base del archivo cargado (ej. "mi_archivo")
            base_name = os.path.basename(self.loaded_filepath)
            base_name = os.path.splitext(base_name)[0]
        else:
            base_name = "datos_analizados" # Nombre de fallback

        num_vars = self.selected_variable_count
        
        # Crear el nombre de archivo sugerido (sin extensión)
        suggested_filename = f"{base_name}_ACP_{num_vars}Variables"
        # ---------------------------------------------------

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx", # La extensión se añade si no se escribe
            filetypes=[("Archivos Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
            title="Guardar TODOS los datos de PCA como...",
            initialfile=suggested_filename # <-- AÑADIR EL NOMBRE SUGERIDO
        )

        if not filepath:
            return

        try:
            # --- 1. Crear el DataFrame con todos los PCs ---
            pc_names = [f"PC{i+1}" for i in range(n_comps_to_save)]
            data_to_save = self.data_transformed[:, :n_comps_to_save]
            df_to_save = pd.DataFrame(data=data_to_save, columns=pc_names)

            # --- 2. AÑADIR LA SUMA ---
            df_to_save['COMPONENTE_SUMA'] = df_to_save.sum(axis=1)

            # 3. Guardar en Excel
            df_to_save.to_excel(filepath, index=False, engine='openpyxl')
            messagebox.showinfo("Éxito", f"TODOS los {n_comps_to_save} componentes PCA (y la columna 'COMPONENTE_SUMA') guardados exitosamente en:\n{filepath}")

        except ImportError:
             messagebox.showerror("Error de Librería", "Para guardar como .xlsx, necesitas instalar la librería 'openpyxl'.\n\nEjecuta en tu terminal: pip install openpyxl")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"Ocurrió un error al guardar el archivo:\n{e}")

    # --- (Las funciones de mostrar matriz no cambian) ---

    def mostrar_ventana_matriz_original(self):
        """
        Crea una nueva ventana (Toplevel) para mostrar la matriz de covarianza ORIGINAL.
        """
        if self.cov_matrix_df is None:
            messagebox.showwarning("Sin Datos", "La matriz de covarianza original aún no ha sido calculada. Ejecute el análisis primero.")
            return

        self._crear_ventana_generica_matriz(
            titulo="Matriz de Varianza-Covarianza (Original)",
            label_text="Matriz de Varianza-Covarianza (Datos Originales)",
            matrix_df=self.cov_matrix_df
        )

    def mostrar_ventana_matriz_pca(self):
        """
        Crea una nueva ventana (Toplevel) para mostrar la matriz de covarianza de PCA.
        """
        if self.pca_cov_matrix_df is None:
            messagebox.showwarning("Sin Datos", "La matriz de covarianza PCA aún no ha sido calculada. Ejecute el análisis primero.")
            return

        self._crear_ventana_generica_matriz(
            titulo="Matriz de Covarianza PCA (Nuevas Variables)",
            label_text="Matriz de Covarianza (Nuevas Variables PCA)",
            matrix_df=self.pca_cov_matrix_df
        )

    def _crear_ventana_generica_matriz(self, titulo, label_text, matrix_df):
        """
        Función auxiliar para crear una ventana modal genérica para mostrar una matriz.
        """
        win_matriz = tk.Toplevel(self)
        win_matriz.title(titulo)
        win_matriz.geometry("900x500")
        win_matriz.configure(bg=self.BG_COLOR)
        win_matriz.minsize(400, 300)

        lbl_title_matriz = tk.Label(win_matriz,
                                      text=label_text,
                                      font=self.TITLE_FONT,
                                      bg=self.BG_COLOR,
                                      fg=self.FG_COLOR)
        lbl_title_matriz.pack(pady=(20, 10))

        bottom_frame_matriz = tk.Frame(win_matriz, bg=self.BG_COLOR)
        bottom_frame_matriz.pack(side="bottom", fill="x", pady=10)

        text_frame_matriz = tk.Frame(win_matriz, bg=self.BG_COLOR)
        text_frame_matriz.pack(fill="both", expand=True, padx=20, pady=10)

        xscrollbar = tk.Scrollbar(text_frame_matriz, orient=tk.HORIZONTAL)
        yscrollbar = tk.Scrollbar(text_frame_matriz, orient=tk.VERTICAL)

        txt_matriz = tk.Text(text_frame_matriz,
                             wrap=tk.NONE,
                             font=self.RESULT_FONT,
                             bg=self.TEXT_BG,
                             fg=self.FG_COLOR,
                             relief=tk.FLAT,
                             borderwidth=0,
                             xscrollcommand=xscrollbar.set,
                             yscrollcommand=yscrollbar.set)

        xscrollbar.config(command=txt_matriz.xview)
        yscrollbar.config(command=txt_matriz.yview)

        text_frame_matriz.grid_rowconfigure(0, weight=1)
        text_frame_matriz.grid_columnconfigure(0, weight=1)

        txt_matriz.grid(row=0, column=0, sticky="nsew")
        yscrollbar.grid(row=0, column=1, sticky="ns")
        xscrollbar.grid(row=1, column=0, sticky="ew")

        try:
            matrix_string = matrix_df.to_string(float_format="%.4f")
        except AttributeError:
             matrix_string = str(matrix_df)

        txt_matriz.insert(tk.END, matrix_string)
        txt_matriz.configure(state='disabled')

        btn_close = tk.Button(bottom_frame_matriz,
                               text="Cerrar",
                               font=self.BUTTON_FONT,
                               bg=self.BTN_EXIT_BG,
                               fg=self.FG_COLOR,
                               command=win_matriz.destroy,
                               relief=tk.FLAT,
                               padx=15,
                               pady=10,
                               bd=0)
        btn_close.pack()

        win_matriz.transient(self)
        win_matriz.grab_set()
        self.wait_window(win_matriz)

    # --- NUEVOS MÉTODOS PARA COMBINACIONES ---
    def abrir_ventana_combinaciones(self):
        """Abre una ventana emergente y calcula las mejores combinaciones."""
        if self.data_raw is None:
            return

        # Crear ventana
        win_combo = tk.Toplevel(self)
        win_combo.title("Mejores Combinaciones de Variables")
        win_combo.geometry("800x600")
        win_combo.configure(bg=self.BG_COLOR)

        # Título
        tk.Label(win_combo, text="Análisis de Retención de Información (Mejores Combinaciones)", 
                 font=self.TITLE_FONT, bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=10)

        # Área de texto
        txt_combo = scrolledtext.ScrolledText(win_combo, wrap=tk.WORD, font=self.RESULT_FONT,
                                              bg=self.TEXT_BG, fg=self.FG_COLOR, relief=tk.FLAT)
        txt_combo.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Configurar tags de color
        txt_combo.tag_config('header', font=self.FINAL_FONT, foreground="#FF9800") # Naranja
        txt_combo.tag_config('var_list', foreground="#4CAF50") # Verde

        # --- Frame para los botones (Cerrar y Descargar) ---
        btn_frame = tk.Frame(win_combo, bg=self.BG_COLOR)
        btn_frame.pack(pady=10)

        # Botón Descargar Reporte
        tk.Button(btn_frame, text="Descargar Informe (.txt)", font=self.BUTTON_FONT, 
                  bg="#2196F3", fg=self.FG_COLOR, # Azul
                  activebackground="#1976D2", activeforeground=self.FG_COLOR,
                  command=lambda: self.descargar_reporte(txt_combo), 
                  relief=tk.FLAT).pack(side="left", padx=10)

        # Botón Cerrar
        tk.Button(btn_frame, text="Cerrar", font=self.BUTTON_FONT, 
                  bg=self.BTN_EXIT_BG, fg=self.FG_COLOR,
                  command=win_combo.destroy, relief=tk.FLAT).pack(side="left", padx=10)

        # Ejecutar cálculo
        self.after(100, lambda: self.calcular_mejores_combinaciones(txt_combo))
        
    def descargar_reporte(self, text_widget):
        """Guarda el contenido del widget de texto en un archivo .txt automáticamente."""
        content = text_widget.get("1.0", tk.END).strip()
        
        if not content or "Calculando..." in content:
            messagebox.showwarning("Espera", "Espera a que termine el cálculo antes de descargar.")
            return

        folder_name = "Mejores Combinaciones"
        
        # 1. Crear carpeta si no existe
        if not os.path.exists(folder_name):
            try:
                os.makedirs(folder_name)
            except OSError as e:
                messagebox.showerror("Error", f"No se pudo crear la carpeta: {e}")
                return

        # 2. Obtener nombre base del archivo CSV analizado
        if self.loaded_filepath:
            base_name = os.path.basename(self.loaded_filepath)
            base_name = os.path.splitext(base_name)[0] # Quitar extensión .csv
        else:
            base_name = "datos_desconocidos"

        # 3. Calcular el número del reporte
        # Buscamos archivos que empiecen con "Reporte_No_" y contengan el nombre base
        current_files = [f for f in os.listdir(folder_name) 
                         if f.startswith("Reporte_No_") and base_name in f and f.endswith(".txt")]
        
        next_number = len(current_files) + 1
        
        # 4. Construir nombre final
        filename = f"Reporte_No_{next_number}_{base_name}.txt"
        full_path = os.path.join(folder_name, filename)

        # 5. Guardar archivo
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            messagebox.showinfo("Informe Guardado", f"Se ha guardado el reporte exitosamente en:\n\n{full_path}")
            
            # Opcional: Abrir la carpeta automáticamente para el usuario
            # os.startfile(folder_name) 
            
        except Exception as e:
            messagebox.showerror("Error al guardar", f"No se pudo guardar el archivo:\n{e}")

    def calcular_mejores_combinaciones(self, text_widget):
        """
        Lógica de fuerza bruta: Prueba combinaciones de k variables y encuentra
        cuál grupo tiene la mayor varianza explicada en su primera componente.
        """
        text_widget.insert(tk.END, "Calculando... por favor espere.\n\n")
        self.update_idletasks()

        # 1. Obtener variables disponibles (las que están chequeadas en el sidebar)
        selected_source_cols = []
        for i, var in enumerate(self.column_vars):
            if var.get():
                selected_source_cols.append(self.all_column_names[i])

        n_total = len(selected_source_cols)
        
        if n_total < 2:
            text_widget.insert(tk.END, "Error: Necesitas seleccionar al menos 2 variables en el panel lateral para hacer combinaciones.")
            return

        # Limpiar datos base
        try:
            data_clean = self.data_raw[selected_source_cols].copy()
            data_clean = data_clean.select_dtypes(include=[np.number]).dropna()
        except Exception as e:
            text_widget.insert(tk.END, f"Error en los datos: {e}")
            return

        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Analizando combinaciones posibles desde {n_total} variables base...\n")
        text_widget.insert(tk.END, "-"*60 + "\n", 'info')

        # 2. Bucle de 2 a 9 variables
        max_k = min(9, n_total) # No podemos buscar 9 si solo tienes 5 variables
        
        for k in range(2, max_k + 1):
            best_variance = -1.0
            best_cols = None
            
            # Generar todas las combinaciones de tamaño k
            # ADVERTENCIA: Si n_total es muy grande (>20), esto puede tardar.
            combos = list(combinations(selected_source_cols, k))
            
            if len(combos) > 5000:
                 text_widget.insert(tk.END, f"Salatando k={k} (demasiadas combinaciones: {len(combos)})...\n")
                 continue

            for cols in combos:
                # Subconjunto de datos
                sub_data = data_clean[list(cols)]
                
                # Estandarizar
                scaler = StandardScaler()
                sub_scaled = scaler.fit_transform(sub_data)
                
                # PCA (solo necesitamos ver la varianza del primer componente para evaluar "fuerza")
                pca = PCA(n_components=1)
                pca.fit(sub_scaled)
                
                var_ratio = pca.explained_variance_ratio_[0]
                
                if var_ratio > best_variance:
                    best_variance = var_ratio
                    best_cols = cols

            # 3. Imprimir resultado para este k
            if best_cols:
                porcentaje = best_variance * 100
                text_widget.insert(tk.END, f"\nMejores {k} Variables:\n", 'header')
                text_widget.insert(tk.END, f"Retención (PC1): {porcentaje:.2f}%\n")
                text_widget.insert(tk.END, f"Variables: {', '.join(best_cols)}\n", 'var_list')
                text_widget.see(tk.END)
                self.update_idletasks() # Mantiene la UI viva

        text_widget.insert(tk.END, "\n" + "="*60 + "\nAnálisis Finalizado.")


if __name__ == "__main__":
    app = AppPCA()
    app.mainloop()