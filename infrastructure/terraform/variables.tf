variable "catalog_names" {
  type    = set(string)
  default = ["marketplace_dev", "marketplace_staging", "marketplace_prod"]
}
variable "storage_root_by_catalog" {
  description = "Cloud-specific managed storage URL for each catalog"
  type        = map(string)
}
