output "catalogs" {
  value = sort([for catalog in databricks_catalog.marketplace : catalog.name])
}
