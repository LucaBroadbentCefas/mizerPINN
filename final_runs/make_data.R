# ------------------------------------------------------------
# Final observation datasets
# ------------------------------------------------------------

input_file <- "validation/fixtures/pde_multispecies/observations.csv"
out_dir    <- "final_runs/observations/final_runs"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

obs <- read.csv(input_file)

# Never alter this object after loading
obs_original <- obs


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

check_obs <- function(x, name) {

  required <- c(
    "obs_type", "species_idx", "gear_idx",
    "t_start", "t_end", "value", "cv"
  )

  stopifnot(all(required %in% names(x)))
  stopifnot(nrow(x) > 0)

  stopifnot(all(is.finite(x$species_idx)))
  stopifnot(all(x$species_idx == floor(x$species_idx)))

  stopifnot(all(is.finite(x$t_start)))
  stopifnot(all(is.finite(x$t_end)))

  stopifnot(all(is.finite(x$value)))
  stopifnot(all(x$value > 0))

  stopifnot(all(is.finite(x$cv)))
  stopifnot(all(x$cv > 0))

  catch_rows <- x$obs_type == "catch_gear"
  stopifnot(all(is.finite(x$gear_idx[catch_rows])))

  cat(
    name,
    ": n =", nrow(x),
    "| years =", length(unique(x$t_start)),
    "| species =", length(unique(x$species_idx)),
    "| year range =", min(x$t_start), "-", max(x$t_start),
    "\n"
  )
}


save_obs <- function(x, filename) {
  check_obs(x, filename)

  write.csv(
    x,
    file.path(out_dir, filename),
    row.names = FALSE
  )
}


# ------------------------------------------------------------
# 0. Exact baseline
# ------------------------------------------------------------

# file.copy avoids even reading/writing the baseline file again
file.copy(
  input_file,
  file.path(out_dir, "perfect.csv"),
  overwrite = TRUE
)


# ------------------------------------------------------------
# 1. Every third year
#
# Keeps years:
# 0, 3, 6, 9, 12, ...
# ------------------------------------------------------------

every_3yr <- obs_original[
  obs_original$t_start %% 3 == 0,
]

save_obs(
  every_3yr,
  "every_3yr.csv"
)


# ------------------------------------------------------------
# 2. Remove years 10-20
#
# Removes annual observations starting:
# 10, 11, ..., 19
#
# i.e. interval [10, 20)
# ------------------------------------------------------------

gap_10_20 <- obs_original[
  !(obs_original$t_start >= 10 &
      obs_original$t_start < 20),
]

save_obs(
  gap_10_20,
  "gap_10_20.csv"
)


# ------------------------------------------------------------
# 3. Remove years 20-30
#
# Removes:
# 20, 21, ..., 29
# ------------------------------------------------------------

gap_30_40 <- obs_original[
  !(obs_original$t_start >= 30 &
      obs_original$t_start < 40),
]

save_obs(
  gap_30_40,
  "gap_30_40.csv"
)


# ------------------------------------------------------------
# 4. Remove species
#
# EDIT THESE.
#
# These are zero-based species_idx values because that is what
# the Python observation code uses.
# ------------------------------------------------------------

species_to_remove <- c(
  3L,
  7L,
  11L
)

# One experiment per missing species
for (sp in species_to_remove) {

  x <- obs_original[
    obs_original$species_idx != sp,
  ]

  save_obs(
    x,
    paste0("missing_species_", sp, ".csv")
  )
}

# ------------------------------------------------------------
# 5. Five noise levels
#
# Same random Normal deviations are used for every CV.
# Therefore the ONLY systematic change between datasets is
# the magnitude of the observation noise.
# ------------------------------------------------------------

noise_cvs  <- c(0.1, 0.2, 0.3, 0.4, 0.5)
#noise_seed <- 20260907
noise_seed <- 202609071
set.seed(noise_seed)

# One z value for every observation.
# Reused across all CV levels.
z <- rnorm(nrow(obs_original))


for (cv in noise_cvs) {

  sd_log <- sqrt(log1p(cv^2))

  x <- obs_original

  # Store provenance
  x$value_true  <- obs_original$value
  x$true_cv     <- cv
  x$true_sd_log <- sd_log
  x$noise_seed  <- noise_seed
  x$replicate_id <- 1L

  # log(y_obs) ~ N(log(y_true), sd_log^2)
  x$value <- exp(
    log(obs_original$value) +
      sd_log * z
  )

  # Tell the PINN the TRUE observation uncertainty
  x$cv <- cv

  save_obs(
    x,
    paste0(
      "noise_cv_",
      gsub("\\.", "p", format(cv, nsmall = 1)),
      ".csv"
    )
  )
}


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

files <- list.files(
  out_dir,
  pattern = "\\.csv$",
  full.names = TRUE
)

summary_table <- do.call(
  rbind,
  lapply(files, function(f) {

    x <- read.csv(f)

    data.frame(
      file = basename(f),
      n_obs = nrow(x),
      n_years = length(unique(x$t_start)),
      n_species = length(unique(x$species_idx)),
      min_year = min(x$t_start),
      max_year = max(x$t_start)
    )
  })
)

print(summary_table)

write.csv(
  summary_table,
  file.path(out_dir, "dataset_summary.csv"),
  row.names = FALSE
)


library(shiny)
library(ggplot2)
library(dplyr)
library(tools)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

original_file <- "validation/fixtures/pde_multispecies/observations.csv"
out_dir       <- "final_runs/observations/final_runs"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

original <- read.csv(original_file)

csv_files <- list.files(
  out_dir,
  pattern = "\\.csv$",
  full.names = TRUE
)

# Optional: exclude summary file if present
csv_files <- csv_files[basename(csv_files) != "dataset_summary.csv"]

if (length(csv_files) == 0) {
  stop("No CSV files found in out_dir.")
}

dataset_names <- file_path_sans_ext(basename(csv_files))
names(csv_files) <- dataset_names

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

make_segments <- function(df) {
  df %>%
    arrange(species_idx, gear_idx, t_start) %>%
    group_by(species_idx, gear_idx) %>%
    mutate(
      segment = cumsum(
        is.na(lag(t_start)) |
          (t_start - lag(t_start) > 1)
      )
    ) %>%
    ungroup()
}

make_summary <- function(df) {
  data.frame(
    n_obs = nrow(df),
    n_species = length(unique(df$species_idx)),
    n_years = length(unique(df$t_start)),
    min_year = min(df$t_start),
    max_year = max(df$t_start)
  )
}

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

ui <- fluidPage(
  titlePanel("Observation dataset comparison"),

  sidebarLayout(
    sidebarPanel(
      selectInput(
        "dataset",
        "Dataset",
        choices = dataset_names,
        selected = dataset_names[1]
      ),

      fluidRow(
        column(
          6,
          actionButton("prev_dataset", "Previous", width = "100%")
        ),
        column(
          6,
          actionButton("next_dataset", "Next", width = "100%")
        )
      ),

      br(),

      checkboxInput("log_y", "Log y-axis", value = TRUE),
      checkboxInput("show_points", "Show points on modified data", value = TRUE),
      checkboxInput("free_y", "Free y-scale by species", value = TRUE),

      br(),

      helpText("Grey = original observations"),
      helpText("Black = selected dataset"),
      helpText("Modified data are segmented so missing years do not get connected by a line."),

      hr(),

      h4("Selected dataset summary"),
      tableOutput("summary_tbl")
    ),

    mainPanel(
      h4(textOutput("dataset_title")),
      plotOutput("obs_plot", height = "850px")
    )
  )
)

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------

server <- function(input, output, session) {

  current_index <- reactiveVal(1)

  observe({
    updateSelectInput(
      session,
      "dataset",
      selected = dataset_names[current_index()]
    )
  })

  observeEvent(input$dataset, {
    idx <- match(input$dataset, dataset_names)
    if (!is.na(idx)) current_index(idx)
  }, ignoreInit = TRUE)

  observeEvent(input$prev_dataset, {
    idx <- current_index() - 1
    if (idx < 1) idx <- length(dataset_names)
    current_index(idx)
  })

  observeEvent(input$next_dataset, {
    idx <- current_index() + 1
    if (idx > length(dataset_names)) idx <- 1
    current_index(idx)
  })

  selected_data <- reactive({
    req(input$dataset)
    read.csv(csv_files[[input$dataset]])
  })

  selected_data_segmented <- reactive({
    make_segments(selected_data())
  })

  original_segmented <- reactive({
    make_segments(original)
  })

  output$dataset_title <- renderText({
    paste("Viewing:", input$dataset)
  })

  output$summary_tbl <- renderTable({
    make_summary(selected_data())
  }, rownames = FALSE)

  output$obs_plot <- renderPlot({
    new_df <- selected_data_segmented()
    old_df <- original_segmented()

    facet_scales <- if (isTRUE(input$free_y)) "free_y" else "fixed"

    p <- ggplot() +

      # Original data in grey
      geom_line(
        data = old_df,
        aes(
          x = t_start,
          y = value,
          group = interaction(species_idx, gear_idx, segment)
        ),
        colour = "grey70",
        linewidth = 0.8
      ) +

      # Modified data in black
      geom_line(
        data = new_df,
        aes(
          x = t_start,
          y = value,
          group = interaction(species_idx, gear_idx, segment)
        ),
        linewidth = 1
      ) +

      facet_wrap(~ species_idx, scales = facet_scales) +

      labs(
        x = "Year",
        y = "Observed value",
        title = "Original vs modified observations",
        subtitle = "Grey = original, black = selected dataset"
      ) +

      theme_bw()

    if (isTRUE(input$show_points)) {
      p <- p +
        geom_point(
          data = new_df,
          aes(
            x = t_start,
            y = value
          ),
          size = 1
        )
    }

    if (isTRUE(input$log_y)) {
      p <- p + scale_y_log10()
    }

    p
  })
}

# ------------------------------------------------------------
# Run app
# ------------------------------------------------------------

shinyApp(ui, server)
