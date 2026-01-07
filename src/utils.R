


## ---------- Plotting ---------- ##

empty_plot <- function() {
  plot(
    NULL, xlab = "", ylab = "", 
    xaxt = "n", yaxt = "n", 
    xlim = c(0, 10), 
    ylim = c(0, 10),
    bty = "n"
  )
}


plot_funs <- function(
    fun_list,
    p_grid,
    ylim,
    colors = NULL,
    widths = NULL,
    types = NULL,
    color_width_type_labels = NULL
) {
  
  n_funs <- length(fun_list)
  
  ## ---- Defaults & validation ---- ##
  if (is.null(colors)) {
    colors <- rep("black", n_funs)
  } else if (length(colors) != n_funs) {
    stop("Length of colors must match length of fun_list.")
  }
  
  if (is.null(widths)) {
    widths <- rep(1, n_funs)
  } else if (length(widths) != n_funs) {
    stop("Length of widths must match length of fun_list.")
  }
  
  if (is.null(types)) {
    types <- rep(1, n_funs)
  } else if (length(types) != n_funs) {
    stop("Length of types must match length of fun_list.")
  }
  
  ## ---- Plot ---- ##
  par(mfrow = c(1, 1))
  plot(
    NULL,
    xlim = range(p_grid),
    ylim = ylim,
    xlab = "p",
    ylab = ""
  )
  
  for (i in seq_len(n_funs)) {
    lines(
      p_grid,
      fun_list[[i]],
      col = colors[i],
      lwd = widths[i],
      lty = types[i]
    )
  }
  
  ## ---- Legend ---- ##
  if (!is.null(color_width_type_labels)) {
    
    required_cols <- c("color", "width", "type", "label")
    if (!all(required_cols %in% names(color_width_type_labels))) {
      stop("color_width_type_labels must have columns: color, width, type, label.")
    }
    
    legend(
      "topleft",
      legend = color_width_type_labels$label,
      col    = color_width_type_labels$color,
      lwd    = color_width_type_labels$width,
      lty    = color_width_type_labels$type,
      bty    = "n"
    )
  }
}


plot_embeddings <- function(
    emb,
    colors = NULL,
    shapes = NULL,
    sizes = NULL,
    color_shape_size_labels = NULL,
    stats = TRUE
) {
  K <- ncol(emb)
  N <- nrow(emb)
  
  ## ---- Color handling ---- ##
  if (is.null(colors)) {
    point_cols <- rep("black", N)
    use_legend <- FALSE
  } else {
    if (length(colors) != N) {
      stop("Length of colors must match number of rows in emb.")
    }
    point_cols <- colors
    use_legend <- TRUE
  }
  
  ## ---- Shape handling ---- ##
  if (is.null(shapes)) {
    point_shapes <- rep(19, N)
  } else {
    if (length(shapes) != N) {
      stop("Length of shapes must match number of rows in emb.")
    }
    point_shapes <- shapes
  }
  
  ## ---- Size handling ---- ##
  if (is.null(sizes)) {
    point_sizes <- rep(1, N)
  } else {
    if (length(sizes) != N) {
      stop("Length of sizes must match number of rows in emb.")
    }
    point_sizes <- sizes
  }
  
  ## ---- Legend validation ---- ##
  if (use_legend) {
    if (is.null(color_shape_size_labels)) {
      stop("color_shape_size_labels must be provided when colors is not NULL.")
    }
    
    required_cols <- c("color", "shape", "size", "label")
    if (!all(required_cols %in% colnames(color_shape_size_labels))) {
      stop("color_shape_size_labels must have columns: color, shape, size, label.")
    }
  }
  
  par(mfrow = c(K, K))
  
  for (k1 in 1:K) {
    for (k2 in 1:K) {
      
      if (k1 == k2) {
        ## ---- QQ plot ---- ##
        if (stats) {
          pval <- round(shapiro.test(emb[, k1])$p.value, 3)
          main_txt <- paste("k =", k1, "| p =", pval)
        } else {
          main_txt <- paste("k =", k1)
        }
        
        qqnorm(
          emb[, k1],
          main = main_txt,
          col  = point_cols,
          pch  = point_shapes,
          cex  = point_sizes
        )
        qqline(emb[, k1])
        
      } else if (k1 > k2) {
        ## ---- Scatter plot ---- ##
        if (stats) {
          r <- round(cor(emb[, k1], emb[, k2]), 3)
          main_txt <- paste("r =", r)
        } else {
          main_txt <- ""
        }
        
        plot(
          emb[, k1], emb[, k2],
          main = main_txt,
          xlab = paste("k =", k1),
          ylab = paste("k =", k2),
          col  = point_cols,
          pch  = point_shapes,
          cex  = point_sizes
        )
        
      } else {
        empty_plot()
      }
      
      ## ---- Legend (draw once) ---- ##
      if (use_legend && k1 == 1 && k2 == 1) {
        legend(
          "topleft",
          legend = color_shape_size_labels$label,
          col    = color_shape_size_labels$color,
          pch    = color_shape_size_labels$shape,
          pt.cex = color_shape_size_labels$size,
          bty    = "n"
        )
      }
    }
  }
}



## ---------- Mass and Density Functions ---------- ##

y_to_pmf <- function(y) {
  if (length(y) == 0) {
    stop("y must be non-empty.")
  }
  
  if (any(is.na(y))) {
    stop("y must not contain NA values.")
  }
  
  tab <- table(y)
  
  pmf <- data.frame(
    value = as.numeric(names(tab)),
    prob  = as.integer(tab) / length(y),
    row.names = NULL
  )
  
  pmf[order(pmf$value), ]
}

set.seed(12345)
y <- sample(1:10, 100, replace = TRUE)
pmf <- y_to_pmf(y)

plot(
  pmf$value,
  pmf$prob,
  type = "h",
  lwd = 2,
  xlab = "y",
  ylab = "P(Y = y)",
  col = "blue"
)
points(pmf$value, pmf$prob, pch = 19, col = "blue")
